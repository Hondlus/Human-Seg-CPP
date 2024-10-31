#include <Windows.h>
#include <iostream>
#include <vector>
typedef void(*HumansegFunc)(std::wstring, std::string, std::string);

int main()
{
	HINSTANCE hinstLib = LoadLibrary(TEXT("humanseg_dll.dll"));
	if (hinstLib != NULL)
	{
		HumansegFunc humanseg = (HumansegFunc)GetProcAddress(hinstLib, "modnet_init");
		if (humanseg != NULL)
		{
			std::cout << "infer...." << std::endl;
			std::wstring model_path(L"C:\\Users\\hldes\\Desktop\\humanseg\\tmp_file\\modnet.onnx");
			humanseg(model_path, "C:\\Users\\hldes\\Desktop\\humanseg\\tmp_file\\234.jpg",
				"C:\\Users\\hldes\\Desktop\\humanseg\\tmp_file\\result.png");
		}
		else
		{
			std::cout << "¼ÓÔØº¯ÊýÊ§°Ü" << std::endl;
		}
	}
	else
	{
		std::cout << "¼ÓÔØdllÊ§°Ü" << std::endl;
	}
}