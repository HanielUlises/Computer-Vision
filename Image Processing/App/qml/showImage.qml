import QtQuick 2.0
import QtQuick.Controls 2.2
import QtQuick.Layouts 1.3
import QtQuick.Dialogs 1.2
import com.app.frame 1.0

Page {
    anchors.fill: parent

    Rectangle {
        id: bar
        width: parent.width
        height: parent.height * 0.1
        color: "orange"

        Label {
            text: "Show Image"
            color: "white"
            anchors.centerIn: parent
        }
    }

    ColumnLayout {
        id: imageContainer
        width: 640
        height: 480

        anchors.top: bar.bottom
        anchors.topMargin: 20
        anchors.horizontalCenter: parent.horizontalCenter

        Image {
            id: logo
            source: "qrc:/qml/logo.png"

            Layout.preferredWidth: 50
            Layout.preferredHeight: 50

            sourceSize.width: 50
            sourceSize.height: 50

            fillMode: Image.PreserveAspectFit

            anchors {
                top: parent.top
                left: parent.left
                topMargin: 8
                leftMargin: 8
            }
        }


        Frame {
            id: selectedImage
            Layout.preferredWidth: imageContainer.width
            Layout.preferredHeight: imageContainer.height
            visible: false
        }
    }

    RowLayout {
        anchors.top: imageContainer.bottom
        anchors.topMargin: 10
        anchors.horizontalCenter: parent.horizontalCenter
        width: parent.width * 0.6
        spacing: width * 0.2

        Button {
            text: "Open Image"
            Layout.preferredWidth: parent.width * 0.4
            onClicked: imageDialog.open()
        }

        Button {
            text: "Go Back"
            Layout.preferredWidth: parent.width * 0.4
            onClicked: loader.pop()
        }
    }

    FileDialog {
        id: imageDialog
        title: "Choose Image"
        selectExisting: true
        onAccepted: {
            selectedImage.open_image(imageDialog.fileUrl)
            picture.visible = false
            selectedImage.visible = true
        }
    }
}
