CFLAGS  := `pkg-config --cflags --libs opencv4` -std=c++11 -O2
LDFLAGS := `pkg-config --libs opencv4`

all:
	g++ distanceField.cpp -o distanceField $(CFLAGS) $(LDFLAGS)
	g++ disparity.cpp -o disparity $(CFLAGS) $(LDFLAGS)
