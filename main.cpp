#include "include/thread.h"
#include <thread>
int main()
{
    Factory vision;

    std::thread thread1(&Factory::producer,std::ref(vision));
	
    std::thread thread2(&Factory::consumer,std::ref(vision));

    std::thread thread3(&Factory::getdata,std::ref(vision));
    thread1.join();

    thread2.join();

    thread3.join();
}

