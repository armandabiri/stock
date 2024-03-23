#pragma once

#include <chrono>
#include <thread>

inline static double tic(int mode = 0) {
  static double t_start;
  auto now = std::chrono::steady_clock::now();
  if (mode == 0) {
    t_start =
        std::chrono::duration_cast<std::chrono::duration<double>>(now.time_since_epoch()).count();
    return t_start;
  } else {
    auto t_end =
        std::chrono::duration_cast<std::chrono::duration<double>>(now.time_since_epoch()).count();
    auto duration = t_end - t_start;
    return duration;
  }
}

inline void wait(const double& seconds) {
  std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(seconds * 1e3)));
}

inline void sleep(const double& seconds) { wait(seconds); }

inline static double toc() { return tic(1); }

inline static double mtoc() { return toc() * 1e3; }

inline static double utoc() { return toc() * 1e6; }

inline static double ntoc() { return toc() * 1e9; }
