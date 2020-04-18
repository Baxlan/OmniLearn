CXX = g++

RM = rm -rf

CXXFLAGS =  -std=c++17 -O3 -Os -s -pthread -Wall -Wextra -Wunused-macros -Wshadow -Wundef -pedantic -Wconversion -Wno-sign-conversion -Wold-style-cast -Wpointer-arith -Wcast-qual -Wcast-align -Wdouble-promotion -Woverloaded-virtual -Wswitch-default -Wl,--no-as-needed -Wunreachable-code -Wno-deprecated-declarations -I$(INCDIR)
#CXXFLAGS = -std=c++17 -g3 -Wa,-mbig-obj -pthread -Wall -Wextra -Wunused-macros -Wshadow -Wundef -pedantic -Wconversion -Wno-sign-conversion -Wold-style-cast -Wpointer-arith -Wcast-qual -Wcast-align -Wdouble-promotion -Woverloaded-virtual -Wswitch-default -Wl,--no-as-needed -Wunreachable-code -Wno-deprecated-declarations -I$(INCDIR)

NAME = test

BINDIR = bin

SRCDIR = src

INCDIR = include

SRCS =  $(SRCDIR)/Activation.cpp \
$(SRCDIR)/Aggregation.cpp \
$(SRCDIR)/cost.cpp \
$(SRCDIR)/csv.cpp \
$(SRCDIR)/decay.cpp \
$(SRCDIR)/Exception.cpp \
$(SRCDIR)/fileString.cpp \
$(SRCDIR)/Layer.cpp \
$(SRCDIR)/Matrix.cpp \
$(SRCDIR)/metric.cpp \
$(SRCDIR)/Network.cpp \
$(SRCDIR)/NetworkIO.cpp \
$(SRCDIR)/Neuron.cpp \
$(SRCDIR)/Parser.cpp \
$(SRCDIR)/preprocess.cpp \
$(SRCDIR)/main.cpp \


OBJS = $(SRCS:.cpp=.o)


all: $(NAME)

$(NAME): $(OBJS)
	$(CXX) $(OBJS) -o $(BINDIR)/$(NAME) $(CXXFLAGS)

clean:
	$(RM) $(OBJS)

fclean: clean
	$(RM) $(NAME)

re: fclean all

.PHONY: all clean fclean re
