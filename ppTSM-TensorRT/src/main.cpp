#include <windows.h>
#include <memory>
#include "DcVideo.h"


void VideoRecTools()
{
	string engine_path = "./model/ppTSMv2.engine";
	//TSMRecognizer* pptsm = new TSMRecognizer(engine_path);
	std::shared_ptr<TSMRecognizer> pptsm = std::make_shared<TSMRecognizer>(engine_path);
	if (pptsm == nullptr || !pptsm->isInit) {
		std::cout << "�㷨��ʼ��ʧ�ܣ�����ϵ�㷨��Ա" << std::endl;
	}
	std::string videoPath = "./video/example.avi";
	int video_class = pptsm->Infer(videoPath);

	system("pause");
}

int main()
{
	VideoRecTools();
}