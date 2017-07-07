#ifndef BASICTIMER_H
#define BASICTIMER_H

//#include <cstdio>
#include <ctime>
#include <sstream>
#include <string>

class BasicTimer
{
public:
    BasicTimer()
        : TimerCount(0)
    {
        TotalTime.tv_sec  = 0;
        TotalTime.tv_nsec = 0;
    }

    void Start() { clock_gettime(CLOCK_REALTIME, &StartTime); }

    void Stop()
    {
        clock_gettime(CLOCK_REALTIME, &StopTime);
        TimerCount++;
        AddTimes(TotalTime, GetTimeDiff(StartTime, StopTime));
    }

    void Reset()
    {
        TimerCount        = 0;
        TotalTime.tv_sec  = 0;
        TotalTime.tv_nsec = 0;
    }

    unsigned long long int GetCount() const { return TimerCount; }

    float GetLastTimeMs() const
    {
        timespec TimeDiff = GetTimeDiff(StartTime, StopTime);
        return TimespecToMs(TimeDiff);
    }

    float GetAverageTimeMs() const { return TimespecToMs(TotalTime) / TimerCount; }

    float GetTimeSinceLastCheck()
    {
        clock_gettime(CLOCK_REALTIME, &StopTime);
        float time = TimespecToSec(GetTimeDiff(StartTime, StopTime));
        StartTime  = StopTime;
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
    timespec TotalTime;

    unsigned long long int TimerCount;

    static timespec GetTimeDiff(const timespec& start, const timespec& end)
    {
        timespec temp;
        if ((end.tv_nsec - start.tv_nsec) < 0)
        {
            temp.tv_sec  = end.tv_sec - start.tv_sec - 1;
            temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
        }
        else
        {
            temp.tv_sec  = end.tv_sec - start.tv_sec;
            temp.tv_nsec = end.tv_nsec - start.tv_nsec;
        }
        return temp;
    }

    static void AddTimes(timespec& LHS, const timespec& RHS)
    {
        LHS.tv_sec += RHS.tv_sec;
        LHS.tv_nsec += RHS.tv_nsec;
        if (LHS.tv_nsec > 1000000000)
        {
            LHS.tv_sec += 1;
            LHS.tv_nsec -= 1000000000;
        }
    }

    static float TimespecToMs(const timespec& time) { return (time.tv_sec * 1000 + time.tv_nsec / 1000000.f); }

    static float TimespecToSec(const timespec& time) { return TimespecToMs(time) / 1000.f; }
};

#endif // BASICTIMER_H
