#include <wx/image.h>

#include "cApp.h"

wxIMPLEMENT_APP(cApp);

cApp::cApp() {

}

cApp::~cApp() {

}

bool cApp::OnInit() {
    wxInitAllImageHandlers();

    main_frame_1 = new cMain();
    main_frame_1->Show(true);

    return true;
}