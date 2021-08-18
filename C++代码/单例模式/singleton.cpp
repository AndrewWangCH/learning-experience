#include <iostream>
#include <thread>

using namespace std;

//单例类
//这种方式称为Meyers’ Singleton.在简单需求情况下这种方式完全可以, 不过遇到多个单例相互依赖需要按顺序析构的情况就有问题了(但是初始化多个静态类是不会出现问题的).
class Singleton {
public:
	static Singleton& getInstance() { // 静态方法
		static Singleton instance; // 静态对象在getInstance中声明
		return instance;
	}

private:
	Singleton() { ok = 0; ng = 0; std::cout << "Singleton::Singleton()" << std::endl; }
	Singleton(const Singleton&) = delete;
	Singleton& operator=(const Singleton&) = delete;
	~Singleton() { std::cout << "Singleton::~Singleton()" << std::endl; }

public:
	int ok;
	int ng;
};


//使用-测试
int main()
{
	cout << Singleton::getInstance().ok++ << endl;
	cout << "地址: " << &Singleton::getInstance() << endl;
	cout << Singleton::getInstance().ok++ << endl;
	cout << "地址: " << &Singleton::getInstance() << endl;


	for (int i = 0; i < 10; ++i) {
		std::thread([]() {
			cout << "地址: " << &Singleton::getInstance() << endl;
		}).join();
	}
	std::cout << &Singleton::getInstance() << std::endl;

	int ok = Singleton::getInstance().ok;
	for (int i = 0; i < 10; ++i)
	{
		cout << ++Singleton::getInstance().ok << endl;
		cout << --Singleton::getInstance().ng << endl;
	}
	std::cout << &Singleton::getInstance() << std::endl;

	return 0;
}


