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
        id: content_layout
        anchors.top: bar.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        spacing: 10

        Item {
            id: image_area
            Layout.fillWidth: true
            Layout.fillHeight: true

            Image {
                id: logo
                source: "qrc:/qml/logo.png"
                width: 40
                height: 40
                fillMode: Image.PreserveAspectFit
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.leftMargin: 8
                anchors.topMargin: 8
            }

            Frame {
                id: selectedImage
                anchors.fill: parent
                anchors.topMargin: 50
                visible: false
            }
        }

        RowLayout {
            id: button_row
            Layout.fillWidth: true
            Layout.preferredHeight: 60
            spacing: 20

            Button {
                text: "Open Image"
                Layout.fillWidth: true
                onClicked: imageDialog.open()
            }

            Button {
                text: "Smooth"
                Layout.fillWidth: true
                onClicked: {
                    if(!selectedImage.smooth_image())
                        errorAlert.open()
                }
            }

            Button {
                text: "Erode"
                Layout.fillWidth: true
                onClicked: {
                    if(!selectedImage.erode_image())
                        errorAlert.open()
                }
            }

            Button {
                text: "Dilate"
                Layout.fillWidth: true
                onClicked: {
                    if(!selectedImage.dilate_image())
                        errorAlert.open()
                }
            }

            Button {
                text: "Contours"
                Layout.fillWidth: true
                onClicked: {
                    if(!selectedImage.find_cotours())
                        errorAlert.open()
                }
            }

            Button {
                text: "Horizontal Lines"
                Layout.fillWidth: true
                onClicked: {
                    if(!selectedImage.find_horizontal_lines())
                        errorAlert.open()
                }
            }

            Button {
                text: "Vertical Lines"
                Layout.fillWidth: true
                onClicked: {
                    if(!selectedImage.find_vertical_lines())
                        errorAlert.open()
                }
            }

            Button {
                text: "Go Back"
                Layout.fillWidth: true
                onClicked: loader.pop()
            }

            
        }
    }

    FileDialog {
        id: imageDialog
        title: "Choose Image"
        selectExisting: true
        onAccepted: {
            selectedImage.open_image(imageDialog.fileUrl)
            selectedImage.visible = true
        }
    }

    Dialog {
        id: errorAlert
        title: "error"
        Text {
            text: "Please open an image"
        }
        standardButtons: StandardButton.Ok
    }
}
