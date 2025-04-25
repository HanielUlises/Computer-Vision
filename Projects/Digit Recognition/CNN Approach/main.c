#include <gtk/gtk.h>
#include "mnist.h"
#include "cnn.h"

ConvLayer conv_layer;
DenseLayer dense_layer;
MNIST_Data *mnist_data;
uint8_t *mnist_labels;

void update_prediction(GtkWidget *widget, gpointer data) {
    // Example: Load a test image and run prediction
    float *test_image = (float *)malloc(conv_layer.input_size * conv_layer.input_size * sizeof(float));
    int index = rand() % mnist_data->num_images;  // Random image index
    for (int i = 0; i < conv_layer.input_size * conv_layer.input_size; i++) {
        test_image[i] = mnist_data->images[index][i] / 255.0f;  // Normalize image
    }

    float prediction = cnn_predict(&conv_layer, &dense_layer, test_image);

    char prediction_text[256];
    snprintf(prediction_text, sizeof(prediction_text), "Predicted label: %d", (int)prediction);

    GtkWidget *label = gtk_label_new(prediction_text);
    gtk_container_add(GTK_CONTAINER(widget), label);
    gtk_widget_show_all(widget);

    free(test_image);
}

int main(int argc, char *argv[]) {
    gtk_init(&argc, &argv);

    mnist_data = load_mnist_images("train-images.idx3-ubyte");
    mnist_labels = load_mnist_labels("train-labels.idx1-ubyte");

    conv_layer.num_filters = 8;
    conv_layer.filter_size = 3;
    conv_layer.stride = 1;
    conv_layer.input_depth = 1;
    conv_layer.input_size = 28;
    conv_layer.output_size = 26; 
    conv_layer.filters = (float *)malloc(conv_layer.num_filters * conv_layer.filter_size * conv_layer.filter_size * sizeof(float));
    conv_layer.biases = (float *)malloc(conv_layer.num_filters * sizeof(float));
    conv_layer.outputs = (float *)malloc(conv_layer.output_size * conv_layer.output_size * conv_layer.num_filters * sizeof(float));
    conv_layer.d_filters = (float *)malloc(conv_layer.num_filters * conv_layer.filter_size * conv_layer.filter_size * sizeof(float));
    conv_layer.d_biases = (float *)malloc(conv_layer.num_filters * sizeof(float));
    conv_layer.d_outputs = (float *)malloc(conv_layer.output_size * conv_layer.output_size * conv_layer.num_filters * sizeof(float));

    dense_layer.input_size = conv_layer.output_size * conv_layer.output_size * conv_layer.num_filters;
    dense_layer.output_size = 10; 
    dense_layer.weights = (float *)malloc(dense_layer.input_size * dense_layer.output_size * sizeof(float));
    dense_layer.biases = (float *)malloc(dense_layer.output_size * sizeof(float));
    dense_layer.outputs = (float *)malloc(dense_layer.output_size * sizeof(float));
    dense_layer.d_weights = (float *)malloc(dense_layer.input_size * dense_layer.output_size * sizeof(float));
    dense_layer.d_biases = (float *)malloc(dense_layer.output_size * sizeof(float));
    dense_layer.d_outputs = (float *)malloc(dense_layer.output_size * sizeof(float));

    cnn_train(&conv_layer, &dense_layer, (float *)mnist_data->images[0], (float *)mnist_labels, 10, 0.01);

    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "CNN Prediction");
    gtk_container_set_border_width(GTK_CONTAINER(window), 10);

    GtkWidget *button = gtk_button_new_with_label("Predict");
    g_signal_connect(button, "clicked", G_CALLBACK(update_prediction), window);

    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_box_pack_start(GTK_BOX(box), button, TRUE, TRUE, 0);
    gtk_container_add(GTK_CONTAINER(window), box);

    gtk_widget_show_all(window);
    gtk_main();

    // Cleanup
    free(conv_layer.filters);
    free(conv_layer.biases);
    free(conv_layer.outputs);
    free(conv_layer.d_filters);
    free(conv_layer.d_biases);
    free(conv_layer.d_outputs);
    free(dense_layer.weights);
    free(dense_layer.biases);
    free(dense_layer.outputs);
    free(dense_layer.d_weights);
    free(dense_layer.d_biases);
    free(dense_layer.d_outputs);
    free_mnist_data(mnist_data);
    free(mnist_labels);

    return 0;
}