#pragma once

#include "cMain.h"

class cApp : public wxApp{
public:
	cApp();
	~cApp();
	virtual bool OnInit();
private:
	cMain* main_frame_1 = nullptr;
};

