#include "cMain.h"

wxBEGIN_EVENT_TABLE(cMain, wxFrame)
EVT_TOOL(wxID_OPEN, cMain::OnButtonUploadClicked)
wxEND_EVENT_TABLE()

cMain::cMain() : wxFrame(nullptr, wxID_ANY, "Digit recognition", wxDefaultPosition, wxSize(800, 600)) {
    // Create a toolbar and add a tool for uploading images
    wxToolBar* toolbar = CreateToolBar(wxTB_HORIZONTAL, wxID_ANY);

    // Correctly reference the icon from the executable's directory
    wxBitmap uploadIcon(wxT("icons/image_upload.png"), wxBITMAP_TYPE_PNG);
    toolbar->AddTool(wxID_OPEN, "Upload Image", uploadIcon);
    toolbar->Realize();

    // Initialize display elements
    m_imageDisplay = new wxStaticBitmap(this, wxID_ANY, wxNullBitmap, wxPoint(170, 10), wxSize(280, 280));
    m_predictionDisplay = new wxStaticText(this, wxID_ANY, "Prediction: None", wxPoint(460, 10), wxSize(300, 50), wxALIGN_CENTER_HORIZONTAL | wxST_NO_AUTORESIZE);

    // Layout using sizers
    wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);
    sizer->Add(toolbar, 0, wxEXPAND);
    sizer->Add(m_imageDisplay, 0, wxALL | wxCENTER, 5);
    sizer->Add(m_predictionDisplay, 0, wxALL | wxCENTER, 5);

    this->SetSizer(sizer);
    this->Layout();
}

void cMain::OnButtonUploadClicked(wxCommandEvent& evt) {
    wxFileDialog openFileDialog(this, _("Open image file"), "", "", "Image files (*.bmp;*.png;*.jpg;*.ico)|*.bmp;*.png;*.jpg;*.ico", wxFD_OPEN | wxFD_FILE_MUST_EXIST);
    if (openFileDialog.ShowModal() == wxID_CANCEL)
        return;

    wxImage image;
    if (image.LoadFile(openFileDialog.GetPath())) {
        if (image.GetWidth() != 28 || image.GetHeight() != 28) {
            wxMessageBox("The image is not 28x28 pixels.", "Error", wxICON_ERROR);
            return;
        }

        unsigned char* imageData = image.GetData();

        std::vector<float> inputVector(28 * 28);
        for (int i = 0; i < 28 * 28; ++i) {
            // Binarization
            // We invert the colors here (black becomes 1, white becomes 0)
            inputVector[i] = (imageData[i * 3] == 0 ? 1.0f : 0.0f);
        }

        // Now inputVector contains the binarized image data in the form expected by the perceptron

        m_imageDisplay->SetBitmap(wxBitmap(image));

    }
    else {
        wxMessageBox("Could not load the image.", "Error", wxICON_ERROR);
    }

    Refresh(); 
}

cMain::~cMain() {
    if (perceptron != nullptr) {
        delete perceptron;
    }
}
