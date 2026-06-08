#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <sys/mman.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>

#include <libcamera/libcamera.h>
#include <libcamera/controls.h>
#include <libcamera/stream.h>
#include <libcamera/framebuffer_allocator.h>

using namespace libcamera;

// Global variables
std::shared_ptr<Camera> camera;
std::unique_ptr<FrameBufferAllocator> allocator;
std::unique_ptr<Request> request;
Stream *stream = nullptr;
std::mutex mtx;
std::condition_variable capture_cv;
bool frame_ready = false;
std::vector<uint8_t> last_frame;
bool running = true;

void requestComplete(Request *req)
{
    if (req->status() == Request::RequestCancelled)
        return;

    const auto &buffers = req->buffers();
    auto it = buffers.find(stream);
    if (it != buffers.end()) {
        FrameBuffer *buffer = it->second;
        const auto &planes = buffer->planes();
        if (!planes.empty()) {
            const FrameBuffer::Plane &plane = planes[0];
            int fd = plane.fd.get();
            size_t size = plane.length;
            
            // Map the buffer
            void *data = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
            if (data != MAP_FAILED) {
                // Copy frame data
                last_frame.assign(static_cast<uint8_t*>(data), 
                                 static_cast<uint8_t*>(data) + size);
                munmap(data, size);
                
                // Signal frame ready
                std::lock_guard<std::mutex> lock(mtx);
                frame_ready = true;
                capture_cv.notify_one();
            }
        }
    }
}

void saveFrame(const std::string& filename, int width, int height) {
    if (last_frame.empty()) return;
    
    // Create OpenCV Mat from RGB data
    cv::Mat image(height, width, CV_8UC3, last_frame.data());
    
    // Convert RGB to BGR (OpenCV uses BGR)
    cv::Mat bgr_image;
    cv::cvtColor(image, bgr_image, cv::COLOR_RGB2BGR);
    
    // Save as JPEG
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
    cv::imwrite(filename, bgr_image, params);
}

int main()
{
    // Start camera manager
    auto cm = std::make_unique<CameraManager>();
    cm->start();

    if (cm->cameras().empty()) {
        std::cerr << "No camera found" << std::endl;
        return 1;
    }
    camera = cm->cameras()[0];
    std::cout << "Using camera: " << camera->id() << std::endl;

    // Acquire and configure
    camera->acquire();
    
    auto config = camera->generateConfiguration({StreamRole::Viewfinder});
    StreamConfiguration &streamConfig = config->at(0);
    
    // CHANGE HERE: Full sensor resolution
    streamConfig.size = Size(4056+8, 3040+8);  // 4:3 aspect ratio
    streamConfig.pixelFormat = formats::RGB888;
    
    config->validate();
    camera->configure(config.get());
    
    std::cout << "Pixel format: " << streamConfig.pixelFormat.toString() << std::endl;
    std::cout << "Size: " << streamConfig.size.width << "x" << streamConfig.size.height << std::endl;

    stream = streamConfig.stream();
    allocator = std::make_unique<FrameBufferAllocator>(camera);
    if (allocator->allocate(stream) < 0) {
        std::cerr << "Failed to allocate buffers" << std::endl;
        return 1;
    }

    // Create request and attach buffer
    request = camera->createRequest();
    const auto &buffers = allocator->buffers(stream);
    if (buffers.empty()) {
        std::cerr << "No buffers available" << std::endl;
        return 1;
    }
    request->addBuffer(stream, buffers[0].get());

    // Enable auto controls
    ControlList controls(camera->controls());
    controls.set(controls::AwbEnable, true);
    controls.set(controls::AeEnable, true);
    request->controls() = controls;

    // Connect callback and start
    camera->requestCompleted.connect(requestComplete);
    camera->start();

    // WARM-UP: Capture 5 frames to let auto-exposure stabilize
    std::cout << "Warming up camera (auto-exposure stabilization)..." << std::endl;
    for (int i = 0; i < 5; ++i) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            frame_ready = false;
        }
        
        camera->queueRequest(request.get());
        
        std::unique_lock<std::mutex> lock(mtx);
        capture_cv.wait(lock, []{ return frame_ready; });
        
        request->reuse(Request::ReuseBuffers);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "Warm-up complete. Starting capture..." << std::endl;

    // Main capture loop - 10 images at 1 FPS
    std::cout << "Capturing 1 image per second for 10 seconds..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        // Reset flag
        {
            std::lock_guard<std::mutex> lock(mtx);
            frame_ready = false;
        }
        
        // Queue request
        camera->queueRequest(request.get());
        
        // Wait for frame
        std::unique_lock<std::mutex> lock(mtx);
        capture_cv.wait(lock, []{ return frame_ready; });
        
        // Save frame with timestamp
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&now_c);
        std::ostringstream oss;
        oss << std::put_time(&tm, "image_%Y%m%d_%H%M%S.jpg");
        std::string filename = oss.str();
        
        // CHANGE HERE: Pass full resolution
        saveFrame(filename, 4056+8, 3040+8);
        std::cout << "Saved: " << filename << std::endl;
        
        // Reuse the same request
        request->reuse(Request::ReuseBuffers);
        
        // Wait 1 second before next capture (except after last)
        if (i < 9) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    // Quick exit to avoid segfault
    std::cout << "Capture complete! Exiting..." << std::endl;
    _exit(0);
}