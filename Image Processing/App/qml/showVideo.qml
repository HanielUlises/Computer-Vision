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
        onUpdateView: selectedImage.set_raw_frame(raw_frame)
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
        height: parent.height * 0.6
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
            visible: true
        }
    }

    RowLayout {
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 20
        anchors.horizontalCenter: parent.horizontalCenter
        width: parent.width * 0.7
        spacing: width * 0.1

        Button {
            text: "Open Video"
            Layout.preferredWidth: parent.width * 0.3
            onClicked: {
                type = 0
                videoDialog.open()
            }
        }

        Button {
            text: "Subtract BG"
            Layout.preferredWidth: parent.width * 0.3
            onClicked: {
                type = 1
                videoDialog.open()
            }
        }

        Button {
            text: "Go Back"
            Layout.preferredWidth: parent.width * 0.3
            onClicked: loader.pop()
        }

    }

    FileDialog {
        id: videoDialog
        title: "Choose Video"
        selectExisting: true
        onAccepted: {
            selectedImage.visible = true
            switch(type) {
                case 0:
                    video_thread.run_capture(videoDialog.fileUrl)
                    break
                case 1:
                    video_thread.run_background_subtraction(videoDialog.fileUrl)
                    break
            }
        }
    }
}
