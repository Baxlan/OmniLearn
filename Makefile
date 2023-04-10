CXX = g++

RM = rm -rf

# release line
CXXFLAGS = -std=c++17 -O3 -Os -s -pthread -fopenmp -Wall -W -Wunused-macros -Wshadow -Wundef -pedantic -Wconversion -Wno-sign-conversion -Wold-style-cast -Wpointer-arith -Wcast-qual -Wcast-align -Wdouble-promotion -Woverloaded-virtual -Wswitch-default -Wunreachable-code -Wno-deprecated-declarations -I$(INCDIR)
# debug line
#CXXFLAGS = -std=c++17 -g3 -Og -Wa,-mbig-obj -pthread -fopenmp -Wall -W -Wunused-macros -Wshadow -Wundef -pedantic -Wconversion -Wno-sign-conversion -Wold-style-cast -Wpointer-arith -Wcast-qual -Wcast-align -Wdouble-promotion -Woverloaded-virtual -Wswitch-default -Wunreachable-code -Wno-deprecated-declarations -I$(INCDIR)

LDFLAGS = -Wl,--no-as-needed

NAME = omnilearn.exe

BINDIR = bin

SRCDIR = src

INCDIR = include

SRCS =  $(SRCDIR)/Activation.cpp \
$(SRCDIR)/Aggregation.cpp \
$(SRCDIR)/cost.cpp \
$(SRCDIR)/csv.cpp \
$(SRCDIR)/scheduler.cpp \
$(SRCDIR)/Exception.cpp \
$(SRCDIR)/Interpreter.cpp \
$(SRCDIR)/Layer.cpp \
$(SRCDIR)/Matrix.cpp \
$(SRCDIR)/metric.cpp \
$(SRCDIR)/Network.cpp \
$(SRCDIR)/NetworkIO.cpp \
$(SRCDIR)/Neuron.cpp \
$(SRCDIR)/optimizer.cpp \
$(SRCDIR)/preprocess.cpp \
$(SRCDIR)/fileString.cpp \
$(SRCDIR)/main.cpp


OBJS = $(SRCS:.cpp=.o)


all: $(NAME)

$(NAME): $(OBJS)
	$(CXX) $(OBJS) -o $(BINDIR)/$(NAME) $(CXXFLAGS) $(LDFLAGS)

clean:
	$(RM) $(OBJS)

fclean: clean
	$(RM) $(NAME)

re: fclean all

.PHONY: all clean fclean re
