CXX = g++

RM = rm -rf

CXXFLAGS = -std=c++17 -s -O2 -Os -pthread -Wall -Wextra -Wunused-macros -Wshadow -Wundef -pedantic -Wpointer-arith -Wcast-qual -Wcast-align -Wdouble-promotion -Wno-float-equal -Woverloaded-virtual -Wno-attributes -Wswitch-default -Wno-deprecated-declarations -Wl,--no-as-needed -I$(INCDIR)

#-Wconversion -Wsign-conversion -Wold-style-cast
#-g3 met tous les symbols de débugage
#-Weffc++
#-Werror
#-O2 -Os augmente la vitesse d'execution et la taille de l'exe
#-Winline
#-s supprime les symbols et les localisations, annule -g
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

SRCS =  $(SRCDIR)/main.cpp \

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
