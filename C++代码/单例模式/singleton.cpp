#include <iostream>
#include <thread>

using namespace std;

//������
//���ַ�ʽ��ΪMeyers�� Singleton.�ڼ�������������ַ�ʽ��ȫ����, ����������������໥������Ҫ��˳���������������������(���ǳ�ʼ�������̬���ǲ�����������).
class Singleton {
public:
	static Singleton& getInstance() { // ��̬����
		static Singleton instance; // ��̬������getInstance������
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


//ʹ��-����
int main()
{
	cout << Singleton::getInstance().ok++ << endl;
	cout << "��ַ: " << &Singleton::getInstance() << endl;
	cout << Singleton::getInstance().ok++ << endl;
	cout << "��ַ: " << &Singleton::getInstance() << endl;


	for (int i = 0; i < 10; ++i) {
		std::thread([]() {
			cout << "��ַ: " << &Singleton::getInstance() << endl;
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


