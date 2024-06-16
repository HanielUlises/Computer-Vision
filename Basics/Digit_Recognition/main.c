#include <gtk/gtk.h>
#include <cairo.h>
#include <string.h>
#include "perceptron.h"
#include "mnist.h"

#define TRAIN_IMAGES_FILE "train-images-idx3-ubyte"
#define TRAIN_LABELS_FILE "train-labels-idx1-ubyte"
#define TEST_IMAGES_FILE  "t10k-images-idx3-ubyte"
#define TEST_LABELS_FILE  "t10k-labels-idx1-ubyte"

#define EPOCHS 1000
#define LEARNING_RATE 0.1
#define MODEL_FILE "perceptron_model.dat"
#define CANVAS_SIZE 28
#define WINDOW_SCALE 10
#define WINDOW_SIZE (CANVAS_SIZE * WINDOW_SCALE)
#define BRUSH_SIZE 1

typedef struct {
    cairo_surface_t *surface;
    GtkWidget *drawing_area;
    Perceptron *perceptron;
    gboolean drawing;
    gdouble last_x;
    gdouble last_y;
} AppWidgets;

void clear_surface(AppWidgets *app_widgets) {
    cairo_t *cr;

    cr = cairo_create(app_widgets->surface);
    cairo_set_source_rgb(cr, 0, 0, 0); // Set background to black
    cairo_paint(cr);
    cairo_destroy(cr);

    gtk_widget_queue_draw(app_widgets->drawing_area);
}

void save_image(AppWidgets *app_widgets, const char *filename) {
    cairo_surface_write_to_png(app_widgets->surface, filename);
}

void process_image(AppWidgets *app_widgets, double *input, int width, int height) {
    cairo_surface_t *surface = app_widgets->surface;
    cairo_surface_flush(surface);
    unsigned char *data = cairo_image_surface_get_data(surface);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * cairo_image_surface_get_stride(surface) + x * 4;
            int gray = (data[index] + data[index + 1] + data[index + 2]) / 3;
            // Normalization as a grayscale value
            input[y * width + x] = gray / 255.0;
        }
    }

    printf("Processed Image Values:\n");
    for (int i = 0; i < width * height; i++) {
        if (i % width == 0) {
            printf("\n");
        }
        printf("%0.2f ", input[i]);
    }
    printf("\n");
}

gboolean on_draw_event(GtkWidget *widget, cairo_t *cr, gpointer data) {
    (void)widget;
    AppWidgets *app_widgets = (AppWidgets *)data;

    cairo_scale(cr, WINDOW_SCALE, WINDOW_SCALE);
    cairo_set_source_surface(cr, app_widgets->surface, 0, 0);
    cairo_paint(cr);
    return FALSE;
}

gboolean on_button_press_event(GtkWidget *widget, GdkEventButton *event, gpointer data) {
    (void)widget;
    AppWidgets *app_widgets = (AppWidgets *)data;

    if (event->button == GDK_BUTTON_PRIMARY) {
        app_widgets->drawing = TRUE;
        app_widgets->last_x = event->x / WINDOW_SCALE;
        app_widgets->last_y = event->y / WINDOW_SCALE;
    }

    return TRUE;
}

gboolean on_button_release_event(GtkWidget *widget, GdkEventButton *event, gpointer data) {
    (void)widget;
    AppWidgets *app_widgets = (AppWidgets *)data;

    if (event->button == GDK_BUTTON_PRIMARY) {
        app_widgets->drawing = FALSE;
    }

    return TRUE;
}

gboolean on_motion_notify_event(GtkWidget *widget, GdkEventMotion *event, gpointer data) {
    (void)widget;
    AppWidgets *app_widgets = (AppWidgets *)data;

    if (app_widgets->drawing) {
        cairo_t *cr = cairo_create(app_widgets->surface);
        cairo_set_source_rgb(cr, 1, 1, 1); 
        cairo_set_line_width(cr, BRUSH_SIZE);
        cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND); 
        cairo_move_to(cr, app_widgets->last_x, app_widgets->last_y);
        cairo_line_to(cr, event->x / WINDOW_SCALE, event->y / WINDOW_SCALE);
        cairo_stroke(cr);
        cairo_destroy(cr);

        app_widgets->last_x = event->x / WINDOW_SCALE;
        app_widgets->last_y = event->y / WINDOW_SCALE;

        gtk_widget_queue_draw(app_widgets->drawing_area);
    }

    return TRUE;
}

void on_clear_button_clicked(GtkButton *button, gpointer data) {
    (void)button; 
    AppWidgets *app_widgets = (AppWidgets *)data;
    clear_surface(app_widgets);
}

void on_predict_button_clicked(GtkButton *button, gpointer data) {
    (void)button;
    AppWidgets *app_widgets = (AppWidgets *)data;
    double input[28 * 28];
    process_image(app_widgets, input, 28, 28);

    save_image(app_widgets, "debug_image.png");

    int predicted = predict_perceptron(app_widgets->perceptron, input);
    g_print("Predicted digit: %d\n", predicted);
}

void train_perceptron_main() {
    printf("Loading MNIST training data...\n");
    MNIST_Data *train_data = load_mnist_images(TRAIN_IMAGES_FILE);
    uint8_t *train_labels = load_mnist_labels(TRAIN_LABELS_FILE);

    printf("Loading MNIST test data...\n");
    MNIST_Data *test_data = load_mnist_images(TEST_IMAGES_FILE);
    uint8_t *test_labels = load_mnist_labels(TEST_LABELS_FILE);

    if (!train_data || !train_labels || !test_data || !test_labels) {
        fprintf(stderr, "Failed to load MNIST data\n");
        return;
    }

    printf("Number of training samples: %d\n", train_data->num_images);

    int input_size = train_data->rows * train_data->cols;
    int num_classes = 10;

    printf("Preparing training data...\n");
    double **train_inputs = (double **)malloc(train_data->num_images * sizeof(double *));
    double **train_targets = (double **)malloc(train_data->num_images * sizeof(double *));
    for (int i = 0; i < train_data->num_images; i++) {
        train_inputs[i] = (double *)malloc(input_size * sizeof(double));
        train_targets[i] = (double *)calloc(num_classes, sizeof(double));

        for (int j = 0; j < input_size; j++) {
            train_inputs[i][j] = train_data->images[i][j] / 255.0;
        }
        train_targets[i][train_labels[i]] = 1.0;
    }

    printf("Creating perceptron model...\n");
    Perceptron *p = create_perceptron(input_size, num_classes);

    printf("Training perceptron model...\n");
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        train_perceptron(p, train_inputs, train_targets, train_data->num_images, 1, LEARNING_RATE);
        if (epoch % 10 == 0) {
            printf("Epoch %d/%d completed\n", epoch + 1, EPOCHS);
        }
    }

    printf("Saving perceptron model to %s...\n", MODEL_FILE);
    save_perceptron(p, MODEL_FILE);

    printf("Evaluating perceptron model...\n");
    int correct = 0;
    #pragma omp parallel for reduction(+:correct)
    for (int i = 0; i < test_data->num_images; i++) {
        double *input = (double *)malloc(input_size * sizeof(double));
        for (int j = 0; j < input_size; j++) {
            input[j] = test_data->images[i][j] / 255.0;
        }
        int predicted = predict_perceptron(p, input);
        if (predicted == test_labels[i]) {
            correct++;
        }
        free(input);
    }

    printf("Accuracy: %.2f%%\n", (correct / (double)test_data->num_images) * 100.0);

    free_perceptron(p);

    for (int i = 0; i < train_data->num_images; i++) {
        free(train_inputs[i]);
        free(train_targets[i]);
    }
    free(train_inputs);
    free(train_targets);

    free_mnist_data(train_data);
    free(train_labels);
    free_mnist_data(test_data);
    free(test_labels);

    printf("Training completed.\n");
}

int main(int argc, char *argv[]) {
    gtk_init(&argc, &argv);

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [train|draw]\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "train") == 0) {
        train_perceptron_main();
        return 0;
    } else if (strcmp(argv[1], "draw") == 0) {
        GtkWidget *window;
        GtkWidget *vbox;
        GtkWidget *hbox;
        GtkWidget *clear_button;
        GtkWidget *predict_button;
        AppWidgets app_widgets;

        window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
        gtk_window_set_title(GTK_WINDOW(window), "Draw a Digit");
        gtk_window_set_default_size(GTK_WINDOW(window), WINDOW_SIZE, WINDOW_SIZE + 50);
        g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

        vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
        gtk_container_add(GTK_CONTAINER(window), vbox);

        app_widgets.drawing_area = gtk_drawing_area_new();
        gtk_widget_set_size_request(app_widgets.drawing_area, WINDOW_SIZE, WINDOW_SIZE);
        gtk_box_pack_start(GTK_BOX(vbox), app_widgets.drawing_area, TRUE, TRUE, 0);

        hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
        gtk_box_pack_start(GTK_BOX(vbox), hbox, FALSE, FALSE, 0);

        clear_button = gtk_button_new_with_label("Clear");
        gtk_box_pack_start(GTK_BOX(hbox), clear_button, TRUE, TRUE, 0);

        predict_button = gtk_button_new_with_label("Predict");
        gtk_box_pack_start(GTK_BOX(hbox), predict_button, TRUE, TRUE, 0);

        app_widgets.surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, CANVAS_SIZE, CANVAS_SIZE);
        clear_surface(&app_widgets);

        app_widgets.drawing = FALSE;
        g_signal_connect(app_widgets.drawing_area, "draw", G_CALLBACK(on_draw_event), &app_widgets);
        g_signal_connect(app_widgets.drawing_area, "button-press-event", G_CALLBACK(on_button_press_event), &app_widgets);
        g_signal_connect(app_widgets.drawing_area, "button-release-event", G_CALLBACK(on_button_release_event), &app_widgets);
        g_signal_connect(app_widgets.drawing_area, "motion-notify-event", G_CALLBACK(on_motion_notify_event), &app_widgets);
        g_signal_connect(clear_button, "clicked", G_CALLBACK(on_clear_button_clicked), &app_widgets);
        g_signal_connect(predict_button, "clicked", G_CALLBACK(on_predict_button_clicked), &app_widgets);

        gtk_widget_set_events(app_widgets.drawing_area, gtk_widget_get_events(app_widgets.drawing_area) | GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK | GDK_POINTER_MOTION_MASK);

        Perceptron* p = create_perceptron(28 * 28, 10);
        load_perceptron(p, MODEL_FILE);
        app_widgets.perceptron = p;

        gtk_widget_show_all(window);
        gtk_main();

        free_perceptron(p);
        cairo_surface_destroy(app_widgets.surface);
    } else {
        fprintf(stderr, "Invalid mode. Use 'train' or 'draw'.\n");
        return 1;
    }

    return 0;
}
