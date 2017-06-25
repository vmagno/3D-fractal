#ifndef BASICTIMER_H
#define BASICTIMER_H

//#include <cstdio>
#include <ctime>
#include <sstream>
#include <string>

class BasicTimer
{
public:
    BasicTimer() : TimerCount(0) {}
    void Start() { clock_gettime(CLOCK_REALTIME, &StartTime); }
    void Stop() { clock_gettime(CLOCK_REALTIME, &StopTime); TimerCount++; }
    void Reset() { TimerCount = 0; }
    unsigned long long int GetCount() const { return TimerCount; }

    float GetAverageTimeMs() const
    {
        timespec TimeDiff = GetTimeDiff(StartTime, StopTime);
        return TimespecToMs(TimeDiff);
        //return 1.f;
        //printf("TimerCount = %llu\n", TimerCount);
    }

    float GetTimeSinceLastCheck()
    {
        clock_gettime(CLOCK_REALTIME, &StopTime);
        float time = TimespecToSec(GetTimeDiff(StartTime, StopTime));
        StartTime = StopTime;
        return time;
    }

    std::string GetAverageTimeMsStr() const
    {
        std::ostringstream ss;
        ss << GetAverageTimeMs();
        return std::string(ss.str());
    }

private:

    timespec StartTime;
    timespec StopTime;
    unsigned long long int TimerCount;

    static timespec GetTimeDiff(const timespec& start, const timespec& end)
    {
        timespec temp;
        if ((end.tv_nsec-start.tv_nsec)<0) {
            temp.tv_sec = end.tv_sec-start.tv_sec-1;
            temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
        } else {
            temp.tv_sec = end.tv_sec-start.tv_sec;
            temp.tv_nsec = end.tv_nsec-start.tv_nsec;
        }
        return temp;
    }

    static float TimespecToMs(const timespec& time)
    {
        return (time.tv_sec * 1000 + time.tv_nsec / 1000000.f);
    }

    static float TimespecToSec(const timespec& time)
    {
        return TimespecToMs(time) / 1000.f;
    }

};

#endif // BASICTIMER_H
