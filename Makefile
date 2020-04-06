CXX = g++

RM = rm -rf

CXXFLAGS = -std=c++17 -g3 -pthread -Wall -Wextra -Wunused-macros -Wshadow -Wundef -pedantic -Wpointer-arith -Wcast-qual -Wcast-align -Wdouble-promotion -Wno-float-equal -Woverloaded-virtual -Wno-attributes -Wswitch-default -Wno-deprecated-declarations -Wl,--no-as-needed -Wa,-mbig-obj -I$(INCDIR)

#-Wconversion -Wsign-conversion -Wold-style-cast
#-g3 met tous les symbols de débugage
#-Weffc++
#-Werror
#-O2 -Os augmente la vitesse d'execution et la taille de l'exe
#-Winline
#-s supprime les symbols et les localisations, annule -g
#-Wa,-mbig-obj autorise les fichiers .o avec plus de 2^16 entries (gros fichiers)
#-Wunreachable-code ==> warning si un bout de code n'est jamais exétuté
#-Wdisabled-optimization ===> warning si le compilateur n'a pas réussi a optimiser un bout de code trop compliqué
#-m8-bit -m16-bit -m32-bit ===> l'alignement des variables se fait sur 8, 16 ou 32 bits (32 par défaut)
#-flto ===> supprime les erreurs de vtable quand des méthodes virtuelles sont déclarées(.h) mais non implémentées(.cpp) (vtable = VMT = virtual method table)
#-DNDEBUG désactive les assertions
#-ftemplate-depth=   ====> augmente la limite de récursion des templates (900 par défaut)

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
$(SRCDIR)/Neuron.cpp \
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
