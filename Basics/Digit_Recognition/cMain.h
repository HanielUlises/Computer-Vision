#include "wx/wx.h"
#include <wx/filedlg.h>
#include <wx/image.h>
#include <wx/bitmap.h>
#include <wx/statbmp.h>

#include "SimplePerceptron.h"

class cMain : public wxFrame {
public:
    cMain();
    ~cMain();

private:
    wxButton* m_btnUpload = nullptr;
    wxStaticBitmap* m_imageDisplay = nullptr;
    wxStaticText* m_predictionDisplay = nullptr;

    void OnButtonUploadClicked(wxCommandEvent& evt);

    SimplePerceptron* perceptron;

    wxDECLARE_EVENT_TABLE();
};
