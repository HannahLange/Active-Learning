#include "stdheader.h"
#include "setting.h"

// ****************** hier enden 61 Zeichen *****************

// Laserintensitxxxt zur Zeit t
double laser_pulse(double t) {
	double period = 1./MULTISHOT_FREQUENCY;
	double intensity = 0.;
	
	for (int i = 0; i < NUMBER_OF_PULSES; i++) {
		intensity += MAX_INTENSITY *
			gauss(PULSE_PEAK, PULSE_WIDTH, t - i * period);
	}
	
	return intensity;
}
