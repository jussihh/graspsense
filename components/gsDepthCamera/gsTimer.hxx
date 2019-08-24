
#include <cstdio>
#include <chrono>
#include <list>
#include <string>

class gsTimer {

private:
	bool enabled;
	std::string timerID;
	std::deque<std::chrono::system_clock::time_point> times;

public:
	gsTimer::gsTimer(std::string atimerID, bool aenabled=true):times(), timerID(atimerID), enabled(aenabled){
		gsTimer::checkpoint(atimerID);
	}
	//Returns the elapsed time since last checkpoint in milliseconds
	int gsTimer::getElapsed() {
		 return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - times.front()).count();
	}
	void gsTimer::checkpoint(std::string message){

		times.push_front(std::chrono::system_clock::now());
		while (times.size() > 2)
			times.pop_back();
		if (enabled) {
			if (times.size() == 1) {
				std::printf((message + " timer started\n").c_str());
			}
			else if (times.size() == 2) {
				std::printf((timerID + ": " + message + ": %i\n").c_str(), std::chrono::duration_cast<std::chrono::microseconds>(times.front() - times.back()).count());
			}
		}
	}

};

class gsFreqTimer {

private:
	bool enabled;
	long n_obs;
	float mean_duration;
	std::string timerID;
	std::list<std::chrono::system_clock::time_point> times;

public:
	gsFreqTimer::gsFreqTimer(std::string atimerID, bool aenabled = true) :times(), timerID(atimerID), enabled(aenabled), n_obs(0), mean_duration(0.0){

	}

	void gsFreqTimer::checkpoint(std::string message = "") {
		if (enabled) {
			times.push_front(std::chrono::system_clock::now());
			while (times.size() > 2)
				times.pop_back();

			if (times.size() == 1) {
				std::printf((timerID + " freq timer started\n").c_str());
			}
			else if (times.size() == 2) {
				n_obs++;
				float duration = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(times.front() - times.back()).count());
				mean_duration = mean_duration/n_obs*(n_obs-1) + duration/n_obs;
				std::printf((timerID + ": " + message + " mean: %4.2f fps, current: %4.2f fps\n").c_str(), 1000000.0/mean_duration, 1000000.0/duration);
			}
		}
	}


};