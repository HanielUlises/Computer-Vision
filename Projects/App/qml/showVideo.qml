import QtQuick 2.0
import QtQuick.Controls 2.2
import QtQuick.Layouts 1.3
import QtQuick.Dialogs 1.2
import com.app.frame 1.0

Page {
    anchors.fill: parent
    property int type: 0

    Connections {
        target: video_thread
        onUpdate_view: selectedImage.set_raw_frame(frame)
    }

    Rectangle {
        id: bar
        width: parent.width
        height: parent.height * 0.1
        color: "orange"

        Label {
            text: "Show Video"
            color: "white"
            anchors.centerIn: parent
        }
    }

    ColumnLayout {
        id: imageContainer
        width: parent.width * 0.5
        height: parent.height * 0.5

        anchors.top: bar.bottom
        anchors.topMargin: 20
        anchors.horizontalCenter: parent.horizontalCenter

        Image {
            id: logo
            source: "qrc:/qml/logo.png"
            Layout.preferredWidth: 50
            Layout.preferredHeight: 50
            fillMode: Image.PreserveAspectFit
            Layout.alignment: Qt.AlignLeft | Qt.AlignTop
        }

        Frame {
            id: selectedImage
            Layout.preferredWidth: imageContainer.width
            Layout.preferredHeight: imageContainer.height
            visible: false
        }
    }

    RowLayout {
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 10
        anchors.horizontalCenter: parent.horizontalCenter
        width: parent.width * 0.7
        spacing: width * 0.1

        Button {
            text: "Open Video"
            onClicked: {
                type = 0
                videoDialog.open()
            }
        }

        Button {
            text: "Subtract BG"
            onClicked: {
                type = 1
                videoDialog.open()
            }
        }

        Button {
            text: "Detect Color"
            onClicked: {
                type = 2
                videoDialog.open()
            }
        }

        Button {
            text: "Open Camera"
            onClicked: {
                selectedImage.visible = true;
                video_thread.run_camera();
            }
        }

        Button {
            text: "Detect Faces"
            onClicked: {
                type = 3;
                videoDialog.open()
            }
        }

        Button {
            text: "Go Back"
            onClicked: loader.pop()
        }

    }

    FileDialog {
        id: videoDialog
        title: "Choose Video"
        onAccepted: {
            selectedImage.visible = true
            switch(type) {
                case 0:
                    video_thread.run_capture(videoDialog.fileUrl);
                    break;
                case 1:
                    video_thread.run_background_subtraction(videoDialog.fileUrl);
                    break;
                case 2:
                    video_thread.run_color_detection(videoDialog.fileUrl);
                    break;
                case 3:
                    video_thread.run_face_detection(videoDialog.fileUrl);
                    break;
            }
        }
    }
}
