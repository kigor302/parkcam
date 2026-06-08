#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <libcamera/libcamera.h>
#include <libcamera/controls.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/stream.h>

using namespace libcamera;
namespace fs = std::filesystem;

static std::atomic<bool> g_running{true};

struct AppConfig {
    int detection_interval_ms = 1000;
    int retention_days = 30;
    int track_timeout_ms = 60000;
    int prune_interval_minutes = 60;
    int capture_width = 1280;
    int capture_height = 720;
    int jpeg_quality = 88;
    bool motion_enabled = true;
    int motion_min_area = 5000;
    int motion_warmup_frames = 10;
    int event_confirm_frames = 1;
    int startup_ignore_ms = 15000;
    bool vehicle_detector_enabled = true;
    float vehicle_confidence = 0.35f;
    float vehicle_min_area_ratio = 0.015f;
    float vehicle_overlap_threshold = 0.10f;
    std::string vehicle_model = "/home/pi/projects/parkcam/models/MobileNetSSD_deploy.caffemodel";
    std::string vehicle_config = "/home/pi/projects/parkcam/models/MobileNetSSD_deploy.prototxt";
    int http_port = 8080;
    std::string data_dir = "/home/pi/projects/parkcam/data";
    std::string python = "/home/pi/projects/parkcam/.venv/bin/python";
    std::string alpr_script = "/home/pi/projects/parkcam/parkcam_alpr.py";
    std::string db_script = "/home/pi/projects/parkcam/parkcam_db.py";
};

static std::mutex g_config_mutex;
static AppConfig g_config;
static std::mutex g_latest_frame_mutex;
static cv::Mat g_latest_frame;
static const std::string kConfigPath = "/home/pi/projects/parkcam/parkcam.conf";

static std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return s.substr(a, b - a + 1);
}

static bool asBool(const std::string &v) {
    return v == "1" || v == "true" || v == "TRUE" || v == "on" || v == "yes";
}

static std::string nowStringForDb() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

static std::string nowStringForFile() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

static std::string shellQuote(const std::string &s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

static std::string runCommand(const std::string &cmd) {
    std::array<char, 256> buf{};
    std::string result;
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) return result;
    while (fgets(buf.data(), static_cast<int>(buf.size()), pipe)) result += buf.data();
    pclose(pipe);
    return result;
}

static void ensureDirs(const AppConfig &cfg) {
    fs::create_directories(cfg.data_dir);
    fs::create_directories(fs::path(cfg.data_dir) / "events");
}

static AppConfig loadConfig() {
    AppConfig cfg;
    std::ifstream in(kConfigPath);
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = trim(line.substr(0, eq));
        std::string v = trim(line.substr(eq + 1));
        try {
            if (k == "detection_interval_ms") cfg.detection_interval_ms = std::max(200, std::stoi(v));
            else if (k == "retention_days") cfg.retention_days = std::max(1, std::stoi(v));
            else if (k == "track_timeout_ms") cfg.track_timeout_ms = std::max(5000, std::stoi(v));
            else if (k == "prune_interval_minutes") cfg.prune_interval_minutes = std::max(1, std::stoi(v));
            else if (k == "capture_width") cfg.capture_width = std::max(320, std::stoi(v));
            else if (k == "capture_height") cfg.capture_height = std::max(240, std::stoi(v));
            else if (k == "jpeg_quality") cfg.jpeg_quality = std::min(100, std::max(40, std::stoi(v)));
            else if (k == "motion_enabled") cfg.motion_enabled = asBool(v);
            else if (k == "motion_min_area") cfg.motion_min_area = std::max(100, std::stoi(v));
            else if (k == "motion_warmup_frames") cfg.motion_warmup_frames = std::max(0, std::stoi(v));
            else if (k == "event_confirm_frames") cfg.event_confirm_frames = std::max(1, std::stoi(v));
            else if (k == "startup_ignore_ms") cfg.startup_ignore_ms = std::max(0, std::stoi(v));
            else if (k == "vehicle_detector_enabled") cfg.vehicle_detector_enabled = asBool(v);
            else if (k == "vehicle_confidence") cfg.vehicle_confidence = std::max(0.05f, std::stof(v));
            else if (k == "vehicle_min_area_ratio") cfg.vehicle_min_area_ratio = std::max(0.0f, std::stof(v));
            else if (k == "vehicle_overlap_threshold") cfg.vehicle_overlap_threshold = std::max(0.0f, std::stof(v));
            else if (k == "vehicle_model") cfg.vehicle_model = v;
            else if (k == "vehicle_config") cfg.vehicle_config = v;
            else if (k == "http_port") cfg.http_port = std::max(1, std::stoi(v));
            else if (k == "data_dir") cfg.data_dir = v;
            else if (k == "python") cfg.python = v;
            else if (k == "alpr_script") cfg.alpr_script = v;
            else if (k == "db_script") cfg.db_script = v;
        } catch (...) {}
    }
    ensureDirs(cfg);
    return cfg;
}

static void saveConfig(const AppConfig &cfg) {
    std::ofstream out(kConfigPath, std::ios::trunc);
    out << "detection_interval_ms=" << cfg.detection_interval_ms << "\n";
    out << "retention_days=" << cfg.retention_days << "\n";
    out << "track_timeout_ms=" << cfg.track_timeout_ms << "\n";
    out << "prune_interval_minutes=" << cfg.prune_interval_minutes << "\n";
    out << "capture_width=" << cfg.capture_width << "\n";
    out << "capture_height=" << cfg.capture_height << "\n";
    out << "jpeg_quality=" << cfg.jpeg_quality << "\n";
    out << "motion_enabled=" << (cfg.motion_enabled ? 1 : 0) << "\n";
    out << "motion_min_area=" << cfg.motion_min_area << "\n";
    out << "motion_warmup_frames=" << cfg.motion_warmup_frames << "\n";
    out << "event_confirm_frames=" << cfg.event_confirm_frames << "\n";
    out << "startup_ignore_ms=" << cfg.startup_ignore_ms << "\n";
    out << "vehicle_detector_enabled=" << (cfg.vehicle_detector_enabled ? 1 : 0) << "\n";
    out << "vehicle_confidence=" << cfg.vehicle_confidence << "\n";
    out << "vehicle_min_area_ratio=" << cfg.vehicle_min_area_ratio << "\n";
    out << "vehicle_overlap_threshold=" << cfg.vehicle_overlap_threshold << "\n";
    out << "vehicle_model=" << cfg.vehicle_model << "\n";
    out << "vehicle_config=" << cfg.vehicle_config << "\n";
    out << "http_port=" << cfg.http_port << "\n";
    out << "data_dir=" << cfg.data_dir << "\n";
    out << "python=" << cfg.python << "\n";
    out << "alpr_script=" << cfg.alpr_script << "\n";
    out << "db_script=" << cfg.db_script << "\n";
}

static std::string dbPath(const AppConfig &cfg) { return (fs::path(cfg.data_dir) / "parkcam.sqlite3").string(); }

static std::string dbCommand(const AppConfig &cfg, const std::string &args) {
    return shellQuote(cfg.python) + " " + shellQuote(cfg.db_script) + " --db " + shellQuote(dbPath(cfg)) + " " + args;
}

static void initDb(const AppConfig &cfg) { runCommand(dbCommand(cfg, "init")); }

struct AlprResult { std::string plate = "Unknown"; double confidence = 0.0; };

static std::string jsonStringValue(const std::string &json, const std::string &key, const std::string &fallback) {
    std::regex re("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch m;
    if (std::regex_search(json, m, re)) return m[1];
    return fallback;
}

static double jsonDoubleValue(const std::string &json, const std::string &key, double fallback) {
    std::regex re("\\\"" + key + "\\\"\\s*:\\s*([0-9.]+)");
    std::smatch m;
    if (std::regex_search(json, m, re)) return std::stod(m[1]);
    return fallback;
}

static int jsonIntValue(const std::string &json, const std::string &key, int fallback) {
    std::regex re("\\\"" + key + "\\\"\\s*:\\s*(\\d+)");
    std::smatch m;
    if (std::regex_search(json, m, re)) return std::stoi(m[1]);
    return fallback;
}

class AlprWorker {
public:
    ~AlprWorker() { stop(); }

    bool start(const AppConfig &cfg) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (running_) return true;
        int toChild[2];
        int fromChild[2];
        if (pipe(toChild) || pipe(fromChild)) return false;
        pid_ = fork();
        if (pid_ == 0) {
            dup2(toChild[0], STDIN_FILENO);
            dup2(fromChild[1], STDOUT_FILENO);
            close(toChild[0]); close(toChild[1]); close(fromChild[0]); close(fromChild[1]);
            execl(cfg.python.c_str(), cfg.python.c_str(), cfg.alpr_script.c_str(), "--worker", static_cast<char *>(nullptr));
            _exit(127);
        }
        close(toChild[0]);
        close(fromChild[1]);
        in_ = fdopen(toChild[1], "w");
        out_ = fdopen(fromChild[0], "r");
        if (!in_ || !out_ || pid_ <= 0) {
            stopLocked();
            return false;
        }
        char line[4096];
        if (!fgets(line, sizeof(line), out_)) {
            stopLocked();
            return false;
        }
        running_ = true;
        return true;
    }

    AlprResult recognize(const AppConfig &cfg, const std::string &imagePath) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_ && !startUnlocked(cfg)) return fallback(cfg, imagePath);
        fprintf(in_, "%s\n", imagePath.c_str());
        fflush(in_);
        char line[4096];
        if (!fgets(line, sizeof(line), out_)) {
            stopLocked();
            return fallback(cfg, imagePath);
        }
        std::string json(line);
        AlprResult r;
        r.plate = jsonStringValue(json, "plate", "Unknown");
        r.confidence = jsonDoubleValue(json, "confidence", 0.0);
        if (r.plate.empty()) r.plate = "Unknown";
        return r;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopLocked();
    }

private:
    bool startUnlocked(const AppConfig &cfg) {
        int toChild[2];
        int fromChild[2];
        if (pipe(toChild) || pipe(fromChild)) return false;
        pid_ = fork();
        if (pid_ == 0) {
            dup2(toChild[0], STDIN_FILENO);
            dup2(fromChild[1], STDOUT_FILENO);
            close(toChild[0]); close(toChild[1]); close(fromChild[0]); close(fromChild[1]);
            execl(cfg.python.c_str(), cfg.python.c_str(), cfg.alpr_script.c_str(), "--worker", static_cast<char *>(nullptr));
            _exit(127);
        }
        close(toChild[0]);
        close(fromChild[1]);
        in_ = fdopen(toChild[1], "w");
        out_ = fdopen(fromChild[0], "r");
        if (!in_ || !out_ || pid_ <= 0) { stopLocked(); return false; }
        char line[4096];
        if (!fgets(line, sizeof(line), out_)) { stopLocked(); return false; }
        running_ = true;
        return true;
    }

    AlprResult fallback(const AppConfig &cfg, const std::string &imagePath) {
        std::string out = runCommand(shellQuote(cfg.python) + " " + shellQuote(cfg.alpr_script) + " " + shellQuote(imagePath));
        AlprResult r;
        r.plate = jsonStringValue(out, "plate", "Unknown");
        r.confidence = jsonDoubleValue(out, "confidence", 0.0);
        if (r.plate.empty()) r.plate = "Unknown";
        return r;
    }

    void stopLocked() {
        if (in_) { fprintf(in_, "__quit__\n"); fflush(in_); fclose(in_); in_ = nullptr; }
        if (out_) { fclose(out_); out_ = nullptr; }
        if (pid_ > 0) { int status = 0; waitpid(pid_, &status, WNOHANG); pid_ = -1; }
        running_ = false;
    }

    std::mutex mutex_;
    FILE *in_ = nullptr;
    FILE *out_ = nullptr;
    pid_t pid_ = -1;
    bool running_ = false;
};

static AlprWorker g_alpr_worker;

static AlprResult recognizePlate(const AppConfig &cfg, const std::string &imagePath) {
    return g_alpr_worker.recognize(cfg, imagePath);
}

static int addEvent(const AppConfig &cfg, const std::string &ts, const std::string &plate, double conf, const std::string &imagePath) {
    std::ostringstream args;
    args << "add --ts " << shellQuote(ts)
         << " --plate " << shellQuote(plate)
         << " --confidence " << std::fixed << std::setprecision(4) << conf
         << " --image " << shellQuote(imagePath);
    std::string out = runCommand(dbCommand(cfg, args.str()));
    return jsonIntValue(out, "id", 0);
}

static void updateEventPlate(const AppConfig &cfg, int eventId, const std::string &plate, double conf) {
    if (eventId <= 0) return;
    std::ostringstream args;
    args << "update --id " << eventId
         << " --plate " << shellQuote(plate)
         << " --confidence " << std::fixed << std::setprecision(4) << conf;
    runCommand(dbCommand(cfg, args.str()));
}

struct AlprJob {
    int eventId = 0;
    std::string imagePath;
    AppConfig cfg;
};

class AsyncAlprUpdater {
public:
    void start() { worker_ = std::thread(&AsyncAlprUpdater::run, this); }
    void enqueue(const AlprJob &job) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(job);
        }
        cv_.notify_one();
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_one();
        if (worker_.joinable()) worker_.join();
    }
private:
    void run() {
        while (true) {
            AlprJob job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&]{ return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty()) break;
                job = queue_.front();
                queue_.pop_front();
            }
            AlprResult alpr = recognizePlate(job.cfg, job.imagePath);
            updateEventPlate(job.cfg, job.eventId, alpr.plate, alpr.confidence);
            std::cerr << "ALPR event=" << job.eventId << " plate=" << alpr.plate << "\\n";
        }
    }
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<AlprJob> queue_;
    std::thread worker_;
    bool stopping_ = false;
};

class CameraCapture {
public:
    bool start(int width, int height) {
        cm_ = std::make_unique<CameraManager>();
        if (cm_->start()) return false;
        if (cm_->cameras().empty()) {
            std::cerr << "No camera found\n";
            return false;
        }
        camera_ = cm_->cameras()[0];
        if (camera_->acquire()) return false;

        config_ = camera_->generateConfiguration({StreamRole::Viewfinder});
        StreamConfiguration &sc = config_->at(0);
        sc.size = Size(width, height);
        sc.pixelFormat = formats::RGB888;
        config_->validate();
        if (camera_->configure(config_.get())) return false;
        width_ = sc.size.width;
        height_ = sc.size.height;
        stream_ = sc.stream();

        allocator_ = std::make_unique<FrameBufferAllocator>(camera_);
        if (allocator_->allocate(stream_) < 0) return false;
        request_ = camera_->createRequest();
        const auto &buffers = allocator_->buffers(stream_);
        if (buffers.empty()) return false;
        request_->addBuffer(stream_, buffers[0].get());
        ControlList controls(camera_->controls());
        controls.set(controls::AwbEnable, true);
        controls.set(controls::AeEnable, true);
        request_->controls() = controls;

        camera_->requestCompleted.connect(this, &CameraCapture::requestComplete);
        if (camera_->start()) return false;
        std::cerr << "Camera started at " << width_ << "x" << height_ << "\n";
        return true;
    }

    bool capture(cv::Mat &bgr) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ready_ = false;
        }
        if (camera_->queueRequest(request_.get())) return false;
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, std::chrono::seconds(5), [&]{ return ready_; })) return false;
        cv::Mat rgb(height_, width_, CV_8UC3, frame_.data());
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        request_->reuse(Request::ReuseBuffers);
        return true;
    }

    void stop() {
        if (camera_) camera_->stop();
        if (camera_) camera_->release();
        if (cm_) cm_->stop();
    }

private:
    void requestComplete(Request *req) {
        if (req->status() == Request::RequestCancelled) return;
        auto it = req->buffers().find(stream_);
        if (it == req->buffers().end()) return;
        FrameBuffer *buffer = it->second;
        if (buffer->planes().empty()) return;
        const auto &plane = buffer->planes()[0];
        void *data = mmap(nullptr, plane.length, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
        if (data == MAP_FAILED) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            frame_.assign(static_cast<uint8_t *>(data), static_cast<uint8_t *>(data) + plane.length);
            ready_ = true;
        }
        munmap(data, plane.length);
        cv_.notify_one();
    }

    std::unique_ptr<CameraManager> cm_;
    std::shared_ptr<Camera> camera_;
    std::unique_ptr<CameraConfiguration> config_;
    std::unique_ptr<FrameBufferAllocator> allocator_;
    std::unique_ptr<Request> request_;
    Stream *stream_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool ready_ = false;
    std::vector<uint8_t> frame_;
};

struct Detection { cv::Rect box; double area = 0; };
struct Track {
    int id = 0;
    cv::Rect box;
    std::chrono::steady_clock::time_point firstSeen;
    std::chrono::steady_clock::time_point lastSeen;
    bool eventTriggered = false;
    int hits = 0;
};

static double iou(const cv::Rect &a, const cv::Rect &b) {
    int inter = (a & b).area();
    int uni = a.area() + b.area() - inter;
    return uni > 0 ? static_cast<double>(inter) / uni : 0.0;
}

class MotionTracker {
public:
    std::vector<Track> update(const cv::Mat &frame, const AppConfig &cfg) {
        auto now = std::chrono::steady_clock::now();
        std::vector<Detection> detections;
        if (cfg.motion_enabled) detections = detectMotion(frame, cfg);
        else detections.push_back({cv::Rect(0, 0, frame.cols, frame.rows), static_cast<double>(frame.cols * frame.rows)});

        std::vector<int> matched(tracks_.size(), 0);
        for (const auto &d : detections) {
            int best = -1;
            double bestScore = 0.0;
            for (size_t i = 0; i < tracks_.size(); ++i) {
                double score = iou(d.box, tracks_[i].box);
                cv::Point dc(d.box.x + d.box.width / 2, d.box.y + d.box.height / 2);
                cv::Point tc(tracks_[i].box.x + tracks_[i].box.width / 2, tracks_[i].box.y + tracks_[i].box.height / 2);
                double dist = cv::norm(dc - tc);
                double maxDim = std::max(d.box.width, d.box.height);
                if (score < 0.05 && dist < maxDim * 0.8) score = 0.15;
                if (score > bestScore) { bestScore = score; best = static_cast<int>(i); }
            }
            if (best >= 0 && bestScore > 0.05) {
                tracks_[best].box = d.box;
                tracks_[best].lastSeen = now;
                tracks_[best].hits++;
                matched[best] = 1;
            } else {
                tracks_.push_back({nextId_++, d.box, now, now, false, 1});
                matched.push_back(1);
            }
        }

        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const Track &t) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(now - t.lastSeen).count() > cfg.track_timeout_ms;
        }), tracks_.end());

        std::vector<Track> newEvents;
        for (auto &t : tracks_) {
            if (!t.eventTriggered && t.hits >= cfg.event_confirm_frames && std::chrono::duration_cast<std::chrono::milliseconds>(now - t.firstSeen).count() >= cfg.detection_interval_ms) {
                t.eventTriggered = true;
                newEvents.push_back(t);
            }
        }
        return newEvents;
    }

private:
    std::vector<Detection> detectMotion(const cv::Mat &frame, const AppConfig &cfg) {
        std::vector<Detection> out;
        cv::Mat small, gray, mask;
        cv::resize(frame, small, cv::Size(), 0.5, 0.5);
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5,5), 0);
        if (!bg_) bg_ = cv::createBackgroundSubtractorMOG2(120, 36.0, false);
        bg_->apply(gray, mask);
        cv::threshold(mask, mask, 200, 255, cv::THRESH_BINARY);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1,-1), 1);
        cv::morphologyEx(mask, mask, cv::MORPH_DILATE, cv::Mat(), cv::Point(-1,-1), 3);
        if (frames_++ < cfg.motion_warmup_frames) return out;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto &c : contours) {
            double area = cv::contourArea(c) * 4.0;
            if (area < cfg.motion_min_area) continue;
            cv::Rect b = cv::boundingRect(c);
            b.x *= 2; b.y *= 2; b.width *= 2; b.height *= 2;
            b &= cv::Rect(0, 0, frame.cols, frame.rows);
            if (b.width < 60 || b.height < 40) continue;
            if (b.area() > frame.cols * frame.rows * 0.85) continue;
            out.push_back({b, area});
        }
        return out;
    }

    int nextId_ = 1;
    int frames_ = 0;
    cv::Ptr<cv::BackgroundSubtractor> bg_;
    std::vector<Track> tracks_;
};

struct VehicleDetection {
    cv::Rect box;
    int classId = 0;
    float confidence = 0.0f;
};

class VehicleDetector {
public:
    bool ensureLoaded(const AppConfig &cfg) {
        if (loaded_) return true;
        if (!fs::exists(cfg.vehicle_model) || !fs::exists(cfg.vehicle_config)) {
            std::cerr << "Vehicle model files not found\n";
            return false;
        }
        try {
            net_ = cv::dnn::readNetFromCaffe(cfg.vehicle_config, cfg.vehicle_model);
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            loaded_ = true;
            std::cerr << "Vehicle detector loaded\n";
        } catch (const cv::Exception &e) {
            std::cerr << "Vehicle detector load failed: " << e.what() << "\n";
            loaded_ = false;
        }
        return loaded_;
    }

    std::vector<VehicleDetection> detect(const cv::Mat &frame, const AppConfig &cfg) {
        std::vector<VehicleDetection> detections;
        if (!cfg.vehicle_detector_enabled) return detections;
        if (!ensureLoaded(cfg)) return detections;
        cv::Mat blob = cv::dnn::blobFromImage(frame, 0.007843, cv::Size(300, 300), cv::Scalar(127.5, 127.5, 127.5), false, false);
        net_.setInput(blob);
        cv::Mat out = net_.forward();
        cv::Mat det(out.size[2], out.size[3], CV_32F, out.ptr<float>());
        int frameArea = frame.cols * frame.rows;
        for (int i = 0; i < det.rows; ++i) {
            float conf = det.at<float>(i, 2);
            if (conf < cfg.vehicle_confidence) continue;
            int classId = static_cast<int>(det.at<float>(i, 1));
            if (classId != 6 && classId != 7 && classId != 14) continue;
            int x1 = std::clamp(static_cast<int>(det.at<float>(i, 3) * frame.cols), 0, frame.cols - 1);
            int y1 = std::clamp(static_cast<int>(det.at<float>(i, 4) * frame.rows), 0, frame.rows - 1);
            int x2 = std::clamp(static_cast<int>(det.at<float>(i, 5) * frame.cols), 0, frame.cols - 1);
            int y2 = std::clamp(static_cast<int>(det.at<float>(i, 6) * frame.rows), 0, frame.rows - 1);
            cv::Rect box(cv::Point(x1, y1), cv::Point(x2, y2));
            box &= cv::Rect(0, 0, frame.cols, frame.rows);
            if (box.area() <= 0) continue;
            if (static_cast<float>(box.area()) / static_cast<float>(frameArea) < cfg.vehicle_min_area_ratio) continue;
            detections.push_back({box, classId, conf});
        }
        return detections;
    }

    bool matchesTrack(const Track &track, const std::vector<VehicleDetection> &vehicles, const AppConfig &cfg) const {
        for (const auto &v : vehicles) {
            double overlap = iou(track.box, v.box);
            cv::Point vc(v.box.x + v.box.width / 2, v.box.y + v.box.height / 2);
            if (overlap >= cfg.vehicle_overlap_threshold || track.box.contains(vc)) return true;
        }
        return false;
    }
private:
    cv::dnn::Net net_;
    bool loaded_ = false;
};

static std::string urlDecode(const std::string &s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            char hex[3] = {s[i+1], s[i+2], 0};
            out += static_cast<char>(std::strtol(hex, nullptr, 16));
            i += 2;
        } else if (s[i] == '+') out += ' ';
        else out += s[i];
    }
    return out;
}

static std::map<std::string, std::string> parseParams(const std::string &q) {
    std::map<std::string, std::string> p;
    size_t start = 0;
    while (start <= q.size()) {
        size_t amp = q.find('&', start);
        std::string part = q.substr(start, amp == std::string::npos ? std::string::npos : amp - start);
        size_t eq = part.find('=');
        if (eq != std::string::npos) p[urlDecode(part.substr(0, eq))] = urlDecode(part.substr(eq + 1));
        if (amp == std::string::npos) break;
        start = amp + 1;
    }
    return p;
}

static std::string htmlEscape(const std::string &s) {
    std::string o;
    for (char c : s) {
        if (c == '&') o += "&amp;";
        else if (c == '<') o += "&lt;";
        else if (c == '>') o += "&gt;";
        else if (c == '"') o += "&quot;";
        else o += c;
    }
    return o;
}

static void sendAll(int fd, const std::string &s) { ::send(fd, s.data(), s.size(), 0); }

static void sendResponse(int fd, const std::string &status, const std::string &type, const std::string &body) {
    std::ostringstream h;
    h << "HTTP/1.1 " << status << "\r\nContent-Type: " << type << "\r\nContent-Length: " << body.size() << "\r\nConnection: close\r\n\r\n";
    sendAll(fd, h.str());
    sendAll(fd, body);
}

static void scheduleRestart() {
    pid_t currentPid = getpid();
    pid_t child = fork();
    if (child == 0) {
        setsid();
        for (int fd = 3; fd < 256; ++fd) close(fd);
        std::string cmd = "sleep 1; kill -TERM " + std::to_string(currentPid) +
                          "; sleep 4; cd /home/pi/projects/parkcam && setsid -f ./parkcam > parkcam.log 2>&1 < /dev/null";
        execl("/bin/sh", "sh", "-c", cmd.c_str(), static_cast<char *>(nullptr));
        _exit(127);
    }
}

static std::string style();
static void sendBytes(int fd, const std::string &status, const std::string &type, const std::vector<uchar> &body) {
    std::ostringstream h;
    h << "HTTP/1.1 " << status << "\r\nContent-Type: " << type << "\r\nContent-Length: " << body.size() << "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    sendAll(fd, h.str());
    if (!body.empty()) ::send(fd, body.data(), body.size(), 0);
}

static bool encodeLatestSnapshot(const AppConfig &cfg, std::vector<uchar> &jpg) {
    cv::Mat frame;
    {
        std::lock_guard<std::mutex> lock(g_latest_frame_mutex);
        if (g_latest_frame.empty()) return false;
        frame = g_latest_frame.clone();
    }
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, std::min(95, cfg.jpeg_quality)};
    return cv::imencode(".jpg", frame, jpg, params);
}

static std::string snapshotHtml() {
    std::ostringstream out;
    out << "<!doctype html><html><head><title>ParkCam Snapshot</title>" << style() << "</head><body>";
    out << "<div class=bar><h2>Snapshot</h2><a href='/'>Main page</a><a href='/config'>Settings</a></div>";
    out << "<div class=panel><button id=take type=button class=primary>Take snapshot</button> <label><input id=auto type=checkbox> Auto refresh</label></div>";
    out << "<img id=img class=preview title='Click to open full-size snapshot' src='/snapshot-image?t=0'>";
    out << "<script>const img=document.getElementById('img'),auto=document.getElementById('auto');function url(){return '/snapshot-image?t='+Date.now()}function snap(){img.src=url()}document.getElementById('take').onclick=snap;img.onclick=()=>window.open(url(),'parkcamSnapshot','popup=yes,scrollbars=yes,resizable=yes,width=1280,height=900');setInterval(()=>{if(auto.checked)snap()},1000);</script>";
    out << "</body></html>";
    return out.str();
}

static bool readFile(const std::string &path, std::string &data) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    data = ss.str();
    return true;
}

static void sendFile(int fd, const std::string &path, const std::string &type) {
    std::string data;
    if (!readFile(path, data)) { sendResponse(fd, "404 Not Found", "text/plain", "not found"); return; }
    std::ostringstream h;
    h << "HTTP/1.1 200 OK\r\nContent-Type: " << type << "\r\nContent-Length: " << data.size() << "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    sendAll(fd, h.str());
    sendAll(fd, data);
}

static std::string style() {
    return "<style>body{font-family:system-ui,Arial;margin:24px;background:#f6f7f9;color:#1d2329}a{color:#075985}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;border-bottom:1px solid #d8dee5;text-align:left}input,select,button{padding:7px;margin:3px}button{cursor:pointer}.primary{background:#075985;color:white;border:1px solid #075985;border-radius:4px}.secondary{background:white;color:#1d2329;border:1px solid #b8c2cc;border-radius:4px}img.preview{max-width:1024px;width:100%;height:auto;border:1px solid #ccd3da;cursor:zoom-in}.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:16px}.panel{background:white;padding:14px;border:1px solid #d8dee5;margin:12px 0}.settings{max-width:760px}.settings th{width:42%;background:#eef2f6}.settings td,.settings th{vertical-align:middle}.settings input,.settings select{min-width:220px}.actions{display:flex;gap:10px;margin-top:12px}</style>";
}

static std::string eventsHtml(const AppConfig &cfg, const std::map<std::string,std::string> &q) {
    int limit = q.count("limit") ? std::max(1, std::min(500, std::stoi(q.at("limit")))) : 50;
    int offset = q.count("offset") ? std::max(0, std::stoi(q.at("offset"))) : 0;
    std::string start = q.count("start") ? q.at("start") : "";
    std::string end = q.count("end") ? q.at("end") : "";
    std::ostringstream args;
    args << "list --limit " << limit << " --offset " << offset;
    if (!start.empty()) args << " --start " << shellQuote(start);
    if (!end.empty()) args << " --end " << shellQuote(end);
    std::string json = runCommand(dbCommand(cfg, args.str()));

    std::ostringstream out;
    out << "<!doctype html><html><head><title>ParkCam Events</title>" << style() << "</head><body>";
    out << "<div class=bar><h2>ParkCam Events</h2><a href='/config'>Settings</a><a href='/snapshot'>Snapshot</a></div>";
    out << "<form class=panel method=get>Start <input name=start value='" << htmlEscape(start) << "' placeholder='YYYY-MM-DD HH:MM:SS'> End <input name=end value='" << htmlEscape(end) << "' placeholder='YYYY-MM-DD HH:MM:SS'> Limit <input name=limit type=number value='" << limit << "' min=1 max=500> <button>Filter</button></form>";
    out << "<table><tr><th>Time</th><th>Plate</th><th>Confidence</th><th>Image</th></tr>";
    std::regex rowRe("\\{\\\"id\\\":(\\d+),\\\"ts\\\":\\\"([^\\\"]*)\\\",\\\"plate\\\":\\\"([^\\\"]*)\\\",\\\"confidence\\\":([^,}]*),\\\"image_path\\\":\\\"([^\\\"]*)\\\"\\}");
    for (std::sregex_iterator it(json.begin(), json.end(), rowRe), endit; it != endit; ++it) {
        std::string id = (*it)[1], ts = (*it)[2], plate = (*it)[3], conf = (*it)[4];
        out << "<tr><td>" << htmlEscape(ts) << "</td><td><b>" << htmlEscape(plate) << "</b></td><td>" << htmlEscape(conf) << "</td><td><a href='/event?id=" << id << "'>open</a></td></tr>";
    }
    out << "</table><div class=bar style='margin-top:14px'>";
    if (offset > 0) out << "<a href='/?limit=" << limit << "&offset=" << std::max(0, offset - limit) << "'>Previous</a>";
    out << "<a href='/?limit=" << limit << "&offset=" << (offset + limit) << "'>Next</a></div>";
    out << "</body></html>";
    return out.str();
}

static std::string eventHtml(const AppConfig &cfg, int id) {
    std::string json = runCommand(dbCommand(cfg, "get --id " + std::to_string(id)));
    std::string ts = jsonStringValue(json, "ts", "");
    std::string plate = jsonStringValue(json, "plate", "Unknown");
    std::string img = jsonStringValue(json, "image_path", "");
    std::ostringstream out;
    out << "<!doctype html><html><head><title>Event</title>" << style() << "</head><body><div class=bar><a href='/'>Events</a><a href='/snapshot'>Snapshot</a></div>";
    out << "<h2>" << htmlEscape(plate) << "</h2><p>" << htmlEscape(ts) << "</p>";
    out << "<img class=preview src='/event-image?id=" << id << "'>";
    out << "</body></html>";
    return out.str();
}

static std::string configHtml(const AppConfig &cfg, const std::string &msg = "") {
    std::ostringstream out;
    out << "<!doctype html><html><head><title>ParkCam Settings</title>" << style() << "</head><body><div class=bar><h2>Settings</h2><a href='/'>Main page</a><a href='/snapshot'>Snapshot</a></div>";
    if (!msg.empty()) out << "<p><b>" << htmlEscape(msg) << "</b></p>";
    out << "<form class=panel method=post>";
    out << "<table class=settings>";
    out << "<tr><th>Description</th><th>Value</th></tr>";
    out << "<tr><td>Detection interval</td><td><input name=detection_interval_ms type=number value='" << cfg.detection_interval_ms << "'> ms</td></tr>";
    out << "<tr><td>Event retention</td><td><input name=retention_days type=number value='" << cfg.retention_days << "'> days</td></tr>";
    out << "<tr><td>Track timeout</td><td><input name=track_timeout_ms type=number value='" << cfg.track_timeout_ms << "'> ms</td></tr>";
    out << "<tr><td>Motion gate</td><td><select name=motion_enabled><option value=1" << (cfg.motion_enabled ? " selected" : "") << ">Enabled</option><option value=0" << (!cfg.motion_enabled ? " selected" : "") << ">Disabled</option></select></td></tr>";
    out << "<tr><td>Motion minimum area</td><td><input name=motion_min_area type=number value='" << cfg.motion_min_area << "'> px</td></tr>";
    out << "<tr><td>Confirm frames</td><td><input name=event_confirm_frames type=number value='" << cfg.event_confirm_frames << "'></td></tr>";
    out << "<tr><td>Startup ignore window</td><td><input name=startup_ignore_ms type=number value='" << cfg.startup_ignore_ms << "'> ms</td></tr>";
    out << "<tr><td>Vehicle detector</td><td><select name=vehicle_detector_enabled><option value=1" << (cfg.vehicle_detector_enabled ? " selected" : "") << ">Enabled</option><option value=0" << (!cfg.vehicle_detector_enabled ? " selected" : "") << ">Disabled</option></select></td></tr>";
    out << "<tr><td>Vehicle confidence</td><td><input name=vehicle_confidence type=number step=0.05 value='" << cfg.vehicle_confidence << "'></td></tr>";
    out << "<tr><td>Vehicle minimum frame area</td><td><input name=vehicle_min_area_ratio type=number step=0.005 value='" << cfg.vehicle_min_area_ratio << "'></td></tr>";
    out << "<tr><td>Vehicle/motion overlap</td><td><input name=vehicle_overlap_threshold type=number step=0.05 value='" << cfg.vehicle_overlap_threshold << "'></td></tr>";
    out << "<tr><td>JPEG quality</td><td><input name=jpeg_quality type=number value='" << cfg.jpeg_quality << "'></td></tr>";
    out << "<tr><td>Camera resolution</td><td><select name=resolution>";
    const std::vector<std::pair<int,int>> resolutions = {{1280,720},{1920,1080},{2560,1440},{3840,2160},{4000,3000}};
    for (const auto &r : resolutions) {
        bool selected = cfg.capture_width == r.first && cfg.capture_height == r.second;
        out << "<option value='" << r.first << "x" << r.second << "'" << (selected ? " selected" : "") << ">" << r.first << "x" << r.second << "</option>";
    }
    out << "</select></td></tr>";
    out << "<tr><td>HTTP port</td><td><input name=http_port type=number value='" << cfg.http_port << "'></td></tr>";
    out << "</table>";
    out << "<div class=actions><button class=secondary name=action value=save>Save</button><button class=primary name=action value=apply>Apply and restart</button></div>";
    out << "<p>Save updates runtime settings where possible. Apply saves settings and restarts ParkCam so camera resolution, HTTP port, and all startup parameters reload.</p>";
    out << "</form></body></html>";
    return out.str();
}

static std::optional<int> getEventImageId(const std::string &query) {
    auto p = parseParams(query);
    if (!p.count("id")) return std::nullopt;
    try { return std::stoi(p["id"]); } catch (...) { return std::nullopt; }
}

static void handleClient(int fd) {
    char buf[8192];
    int n = recv(fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) { close(fd); return; }
    std::string req(buf, n);
    size_t lineEnd = req.find("\r\n");
    std::string line = req.substr(0, lineEnd);
    std::istringstream ls(line);
    std::string method, target, version;
    ls >> method >> target >> version;
    size_t qpos = target.find('?');
    std::string path = qpos == std::string::npos ? target : target.substr(0, qpos);
    std::string query = qpos == std::string::npos ? "" : target.substr(qpos + 1);

    AppConfig cfg;
    { std::lock_guard<std::mutex> lock(g_config_mutex); cfg = g_config; }

    if (method == "POST") {
        size_t headerEnd = req.find("\r\n\r\n");
        std::string body = headerEnd == std::string::npos ? "" : req.substr(headerEnd + 4);
        auto p = parseParams(body);
        if (path == "/config") {
            AppConfig next = cfg;
            try {
                if (p.count("detection_interval_ms")) next.detection_interval_ms = std::max(200, std::stoi(p["detection_interval_ms"]));
                if (p.count("retention_days")) next.retention_days = std::max(1, std::stoi(p["retention_days"]));
                if (p.count("track_timeout_ms")) next.track_timeout_ms = std::max(5000, std::stoi(p["track_timeout_ms"]));
                if (p.count("motion_enabled")) next.motion_enabled = asBool(p["motion_enabled"]);
                if (p.count("motion_min_area")) next.motion_min_area = std::max(100, std::stoi(p["motion_min_area"]));
                if (p.count("event_confirm_frames")) next.event_confirm_frames = std::max(1, std::stoi(p["event_confirm_frames"]));
                if (p.count("startup_ignore_ms")) next.startup_ignore_ms = std::max(0, std::stoi(p["startup_ignore_ms"]));
                if (p.count("vehicle_detector_enabled")) next.vehicle_detector_enabled = asBool(p["vehicle_detector_enabled"]);
                if (p.count("vehicle_confidence")) next.vehicle_confidence = std::max(0.05f, std::stof(p["vehicle_confidence"]));
                if (p.count("vehicle_min_area_ratio")) next.vehicle_min_area_ratio = std::max(0.0f, std::stof(p["vehicle_min_area_ratio"]));
                if (p.count("vehicle_overlap_threshold")) next.vehicle_overlap_threshold = std::max(0.0f, std::stof(p["vehicle_overlap_threshold"]));
                if (p.count("jpeg_quality")) next.jpeg_quality = std::min(100, std::max(40, std::stoi(p["jpeg_quality"])));
                if (p.count("resolution")) {
                    std::smatch m;
                    if (std::regex_match(p["resolution"], m, std::regex("(\\d+)x(\\d+)"))) {
                        next.capture_width = std::stoi(m[1]);
                        next.capture_height = std::stoi(m[2]);
                    }
                }
                if (p.count("capture_width")) next.capture_width = std::max(320, std::stoi(p["capture_width"]));
                if (p.count("capture_height")) next.capture_height = std::max(240, std::stoi(p["capture_height"]));
                if (p.count("http_port")) next.http_port = std::max(1, std::stoi(p["http_port"]));
                ensureDirs(next);
                saveConfig(next);
                { std::lock_guard<std::mutex> lock(g_config_mutex); g_config = next; }
                bool apply = p.count("action") && p["action"] == "apply";
                if (apply) {
                    sendResponse(fd, "200 OK", "text/html", configHtml(next, "Saved. Restarting ParkCam now."));
                    scheduleRestart();
                } else {
                    sendResponse(fd, "200 OK", "text/html", configHtml(next, "Saved"));
                }
            } catch (...) {
                sendResponse(fd, "400 Bad Request", "text/html", configHtml(cfg, "Invalid setting"));
            }
        } else sendResponse(fd, "404 Not Found", "text/plain", "not found");
        close(fd); return;
    }

    if (path == "/" || path == "/events") sendResponse(fd, "200 OK", "text/html", eventsHtml(cfg, parseParams(query)));
    else if (path == "/config") sendResponse(fd, "200 OK", "text/html", configHtml(cfg));
    else if (path == "/snapshot") sendResponse(fd, "200 OK", "text/html", snapshotHtml());
    else if (path == "/snapshot-image") {
        std::vector<uchar> jpg;
        if (encodeLatestSnapshot(cfg, jpg)) sendBytes(fd, "200 OK", "image/jpeg", jpg);
        else sendResponse(fd, "503 Service Unavailable", "text/plain", "snapshot not ready");
    }
    else if (path == "/event") {
        auto id = getEventImageId(query);
        if (id) sendResponse(fd, "200 OK", "text/html", eventHtml(cfg, *id)); else sendResponse(fd, "400 Bad Request", "text/plain", "missing id");
    } else if (path == "/event-image") {
        auto id = getEventImageId(query);
        if (!id) sendResponse(fd, "400 Bad Request", "text/plain", "missing id");
        else {
            std::string json = runCommand(dbCommand(cfg, "get --id " + std::to_string(*id)));
            std::string img = jsonStringValue(json, "image_path", "");
            fs::path root = fs::weakly_canonical(cfg.data_dir);
            fs::path file = fs::weakly_canonical(img);
            if (img.empty() || file.string().find(root.string()) != 0) sendResponse(fd, "404 Not Found", "text/plain", "not found");
            else sendFile(fd, file.string(), "image/jpeg");
        }
    } else sendResponse(fd, "404 Not Found", "text/plain", "not found");
    close(fd);
}

static void webServer(int port) {
    int server = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    bool bound = false;
    for (int attempt = 0; attempt < 30 && g_running; ++attempt) {
        if (bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            bound = true;
            break;
        }
        perror("bind");
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    if (!bound) { close(server); return; }
    if (listen(server, 16) < 0) { perror("listen"); close(server); return; }
    std::cerr << "Web UI listening on http://0.0.0.0:" << port << "\n";
    while (g_running) {
        sockaddr_in client{}; socklen_t len = sizeof(client);
        int fd = accept(server, reinterpret_cast<sockaddr*>(&client), &len);
        if (fd >= 0) std::thread(handleClient, fd).detach();
    }
    close(server);
}

static void signalHandler(int) { g_running = false; }

int main() {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    g_config = loadConfig();
    saveConfig(g_config);
    initDb(g_config);

    AppConfig cfg;
    { std::lock_guard<std::mutex> lock(g_config_mutex); cfg = g_config; }

    std::thread web(webServer, cfg.http_port);
    web.detach();

    CameraCapture cam;
    if (!cam.start(cfg.capture_width, cfg.capture_height)) {
        std::cerr << "Failed to start camera\n";
        return 1;
    }

    MotionTracker tracker;
    VehicleDetector vehicleDetector;
    AsyncAlprUpdater alprUpdater;
    alprUpdater.start();
    auto startedAt = std::chrono::steady_clock::now();
    auto lastPrune = std::chrono::steady_clock::now() - std::chrono::hours(24);
    while (g_running) {
        auto loopStart = std::chrono::steady_clock::now();
        { std::lock_guard<std::mutex> lock(g_config_mutex); cfg = g_config; }

        cv::Mat frame;
        if (!cam.capture(frame)) {
            std::cerr << "Capture timeout\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }

        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, std::min(95, cfg.jpeg_quality)};
        {
            std::lock_guard<std::mutex> lock(g_latest_frame_mutex);
            g_latest_frame = frame.clone();
        }

        auto events = tracker.update(frame, cfg);
        bool startupSettled = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startedAt).count() >= cfg.startup_ignore_ms;
        if (!startupSettled) events.clear();
        std::vector<VehicleDetection> vehicles;
        if (!events.empty() && cfg.vehicle_detector_enabled) vehicles = vehicleDetector.detect(frame, cfg);
        for (const auto &t : events) {
            if (cfg.vehicle_detector_enabled && !vehicleDetector.matchesTrack(t, vehicles, cfg)) {
                std::cerr << "Rejected motion track=" << t.id << " no vehicle\n";
                continue;
            }
            std::string tsDb = nowStringForDb();
            std::string imgPath = (fs::path(cfg.data_dir) / "events" / ("event_" + nowStringForFile() + "_t" + std::to_string(t.id) + ".jpg")).string();
            cv::imwrite(imgPath, frame, params);
            int eventId = addEvent(cfg, tsDb, "Processing", 0.0, imgPath);
            if (eventId > 0) alprUpdater.enqueue({eventId, imgPath, cfg});
            std::cerr << "Event track=" << t.id << " event=" << eventId << " vehicles=" << vehicles.size() << " image=" << imgPath << "\n";
        }

        if (std::chrono::duration_cast<std::chrono::minutes>(loopStart - lastPrune).count() >= cfg.prune_interval_minutes) {
            runCommand(dbCommand(cfg, "prune --days " + std::to_string(cfg.retention_days)));
            lastPrune = loopStart;
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - loopStart).count();
        int sleepMs = std::max(0, cfg.detection_interval_ms - static_cast<int>(elapsed));
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
    }
    alprUpdater.stop();
    g_alpr_worker.stop();
    cam.stop();
    return 0;
}
