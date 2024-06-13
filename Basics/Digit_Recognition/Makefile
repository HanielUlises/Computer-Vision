CC = gcc
CFLAGS = -Wall -Wextra `sdl2-config --cflags`
LDFLAGS = -lm `sdl2-config --libs`
SRCS = main.c imageWriter.c perceptron.c mnist.c
OBJDIR = objects
OBJS = $(patsubst %.c,$(OBJDIR)/%.o,$(SRCS))
EXEC = perceptron

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(EXEC) $(LDFLAGS)

$(OBJDIR)/%.o: %.c
	mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(EXEC)
