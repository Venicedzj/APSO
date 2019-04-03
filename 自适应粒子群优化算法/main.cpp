#include<iostream>
#include<cstdlib>
#include<time.h>
#include<vector>
#include<iomanip>
#include<cmath>
#include<limits>
using namespace std;

struct Particle
{
	vector<double> V;
	vector<double> POS;
	vector<double> pBest;
	double fitness;
	double pBest_ftns;
	double distance;
};

struct GBest
{
	vector<double> POS;
	double fitness;
};

void initPtc();
void APSO();
void get_gBest();
void update_V_POS();
double LDW();
double ranW();
double ConcFDW();
double ConvFDW();
void print();
double getFitnessVal(vector<double> POS);
void calParticlesFitness(vector<double> pos, double & fitness);
double getMeanDistance(Particle ptc);
void calParticlesDistance();
double getEvolutionaryFactor();
int judgeMembership(double evolfactor, int prevstate);
void adjust_c1_c2(int membership, double &c1, double &c2);
void ELS(GBest &gBest);
double generateSigma();
double generateGaussian(double mu, double sigma);

#define PARTICLE_NUM 30		//particle's number
#define V_MAX 6.0			//maximum velocity
#define V_MIN -6.0			//minimum velocity
#define DIMENSION 2		//spatial dimension
#define POS_MAX 100.0		//maximum position coordinate
#define POS_MIN -100.0		//minimum position coordinate
#define wMax 0.9			//maximum weights
#define wMin 0.4			//minimum weights
#define Tmax 200			//maximum iterations number
double c1 = 2.0; 			//individual cognitive coefficient
double c2 = 2.0; 			//social learning coefficient
int T = 1;					//current iterations number
#define t (double)T/Tmax	//evolution time
#define e 2.718281828
int prevState;
vector<Particle> particles(PARTICLE_NUM);
GBest gBest;
inline double random(double a, double b) { return ((double)rand() / RAND_MAX)*(b - a) + a; }
double(*wghtFunc[4])() = { LDW,ranW,ConcFDW,ConvFDW };

/*linear decrement weights*/
double LDW() { return wMax - (wMax - wMin)*t; }
/*random weights*/
double ranW() { return random(0.4, 0.6); }
/*concave function decrement weights*/
double ConcFDW() { return wMax - (wMax - wMin)*t*t; }
/*convex function decrement weights*/
double ConvFDW() { return wMin + (wMax - wMin)*(t - 1)*(t - 1); }

double getWeight(double f) {
	return 1 / (1 + 1.5*pow(e, -2.6*f));
}


int main() {
	srand((int)time(0));
	initPtc();
	while (T <= Tmax) {
		APSO();
		T++;
	}
	return 0;
}

/*steps of the PSO algorithm*/
void APSO() {
	//cout << getEvolutionaryFactor() << endl;
	//print();
	update_V_POS();
	get_gBest();
}

/*initialize each particle's velocity, position, pBest and global best position gBest*/
void initPtc() {
	for (auto &ptc : particles) {
		for (int i = 0; i < DIMENSION; i++) {
			ptc.V.push_back(random(V_MIN, V_MAX));
			ptc.POS.push_back(random(POS_MIN, POS_MAX));
		}
		ptc.pBest = ptc.POS;
		calParticlesFitness(ptc.POS, ptc.fitness);
		calParticlesFitness(ptc.pBest, ptc.pBest_ftns);
	}
	calParticlesDistance();

	int minmark = 0;
	double minfitness = particles[0].fitness;
	for (int i = 0; i < PARTICLE_NUM; i++) {
		if (particles[i].fitness < minfitness) {
			minfitness = particles[i].fitness;
			minmark = i;
		}
	}
	gBest.POS = particles[minmark].POS;
	gBest.fitness = minfitness;

	ELS(gBest);
	prevState = judgeMembership(getEvolutionaryFactor(), 4);
}

/*calculate gBest after updating velocity and position*/
void get_gBest() {
	double minimum = particles[0].fitness;
	int flag = 0;
	for (int i = 0; i < PARTICLE_NUM; i++) {
		/*if (T == 199) {
			for (auto p : particles[i].POS)
				cout << p << " ";
			cout << endl;
		}*/
		if (particles[i].fitness < minimum) {
			flag = i;
			minimum = particles[i].fitness;
		}
	}
	if (particles[flag].fitness < gBest.fitness||T==100) {
		gBest.POS = particles[flag].POS;
		gBest.fitness = particles[flag].fitness;
	}
	ELS(gBest);
	/*if (T == 199) {
		for (auto i : gBest.POS) {
			cout << i << " ";
		}
		cout << endl;
	}*/
	return;
}

/*update particle's velocity, position and pBest*/
void update_V_POS() {
	double f = getEvolutionaryFactor();
	double w = getWeight(f);
	adjust_c1_c2(judgeMembership(f, prevState), c1, c2);
	//cout << c1 << " " << c2<<" "<< prevState<<" ";
	prevState = judgeMembership(f, prevState);
	for (auto &ptc : particles) {
		for (int i = 0; i < DIMENSION; i++) {
			double r1 = random(0, 1), r2 = random(0, 1);
			ptc.V[i] = w*ptc.V[i]
				+ c1*r1*(ptc.pBest[i] - ptc.POS[i])
				+ c2*r2*(gBest.POS[i] - ptc.POS[i]);
			if (ptc.V[i] > V_MAX) ptc.V[i] = V_MAX;
			if (ptc.V[i] < V_MIN) ptc.V[i] = V_MIN;
			ptc.POS[i] = ptc.POS[i] + ptc.V[i];
			if (ptc.POS[i] > POS_MAX) ptc.POS[i] = POS_MAX;
			if (ptc.POS[i] < POS_MIN) ptc.POS[i] = POS_MIN;
		}
		if (ptc.fitness < ptc.pBest_ftns) ptc.pBest = ptc.POS;
		calParticlesFitness(ptc.POS, ptc.fitness);
		calParticlesFitness(ptc.pBest, ptc.pBest_ftns);
	}
	calParticlesDistance();
}

/*using position coordinates as parameter calculate the fitness value by fitness function*/
double getFitnessVal(vector<double> POS) {
	double fitness = 0;
	for (auto pos : POS) {
		if(T<100) fitness += (pos+50)*(pos+50);
		else fitness += (pos-50)*(pos-50);
	}
	return fitness;
}

/*calculate particles' fitness and pBest fitness value*/
void calParticlesFitness(vector<double> pos, double & fitness) {
	fitness = getFitnessVal(pos);
	return;
}

/*calculate particle's mean distance*/
double getMeanDistance(Particle ptc) {
	double sumN = 0;
	for (int i = 0; i < PARTICLE_NUM; i++) {
		double sumD = 0;
		for (int k = 0; k < DIMENSION; k++) {
			sumD += pow(ptc.POS[k] - particles[i].POS[k], 2);
		}
		sumN += sqrt(sumD);
	}
	return sumN / (PARTICLE_NUM - 1);
}

/*calculate all the particles' mean distance*/
void calParticlesDistance() {
	for (auto & ptc : particles) {
		ptc.distance = getMeanDistance(ptc);
	}
}

/*generate the evolutionary factor*/
double getEvolutionaryFactor() {
	double f;
	//the distance of the current best particle as dg
	double bestfitness = particles[0].fitness;
	int dg_index = 0;
	for (int i = 0; i < PARTICLE_NUM; i++) {
		if (particles[i].fitness < bestfitness) {
			dg_index = i;
			bestfitness = particles[i].fitness;
		}
	}
	double dg = particles[dg_index].distance;
	//find dmax and dmin
	double dmax = particles[0].distance;
	double dmin = particles[0].distance;
	for (auto ptc : particles) {
		if (ptc.distance > dmax) {
			dmax = ptc.distance;
		}
		if (ptc.distance < dmin) {
			dmin = ptc.distance;
		}
	}
	f = (dg - dmin) / (dmax - dmin);
	return f;
}

/*judge current membership status set according to previous state*/
int judgeMembership(double evolfactor, int prevstate) {
	if (evolfactor >= 0 && evolfactor <= 0.2) return 3;
	else if (evolfactor >= 0.3 && evolfactor <= 0.4) return 2;
	else if (evolfactor >= 0.6 && evolfactor <= 0.7) return 1;
	else if (evolfactor >= 0.8 && evolfactor <= 1.0) return 4;
	else if (evolfactor > 0.2 && evolfactor < 0.3) {
		switch (prevstate) {
			case 1:
				return 2;
				break;
			case 2:
				return 2;
				break;
			case 3:
				return 3;
				break;
			case 4:
				return 2;
				break;
			default:
				break;
		}
	}
	else if (evolfactor > 0.4 && evolfactor < 0.6) {
		switch (prevstate) {
		case 1:
			return 1;
			break;
		case 2:
			return 2;
			break;
		case 3:
			return 2;
			break;
		case 4:
			return 1;
			break;
		default:
			break;
		}
	}
	else if (evolfactor > 0.7 && evolfactor < 0.8) {
		switch (prevstate) {
		case 1:
			return 1;
			break;
		case 2:
			return 1;
			break;
		case 3:
			return 4;
			break;
		case 4:
			return 4;
			break;
		default:
			break;
		}
	}
	else return 4;
}

/*adjust parameter c1, c2 according to the membership status set*/
void adjust_c1_c2(int membership, double &c1, double &c2) {
	double deta = random(0.05, 0.1);
	switch (membership) {
		case 1:
			c1 += deta;
			c2 -= deta;
			break;
		case 2:
			c1 += 0.5*deta;
			c2 -= 0.5*deta;
			break;
		case 3:
			c1 += 0.5*deta;
			c2 += 0.5*deta;
			break;
		case 4:
			c1 -= deta;
			c2 += deta;
			break;
	}
	if (c1 < 1.5)c1 = 1.5;
	else if (c1>2.5)c1 = 2.5;
	if (c2 < 1.5)c1 = 1.5;
	else if (c2>2.5)c1 = 2.5;
	if (c1 + c2<3.0 || c1 + c2>4.0) {
		c1 = 4.0*(c1 / (c1 + c2));
		c2 = 4.0*(c2 / (c1 + c2));
	}
	return;
}

/*ELS function, push gBest out to a potentially better region*/
void ELS(GBest &gBest) {
	GBest P = gBest;
	int d = (int)random(0, DIMENSION - 1);
	P.POS[d] += (POS_MAX - POS_MIN)*generateGaussian(0, generateSigma());
	P.fitness = getFitnessVal(P.POS);
	if (P.fitness < gBest.fitness) {
		gBest = P;
	}
	else {
		double minimum = particles[0].fitness;
		int flag = 0;
		for (int i = 0; i < PARTICLE_NUM; i++) {
			if (particles[i].fitness < minimum) {
				flag = i;
				minimum = particles[i].fitness;
			}
		}
		particles[flag].fitness = P.fitness;
		particles[flag].POS = P.POS;
		particles[flag].distance = getMeanDistance(particles[flag]);
	}
}

/*generate parameter sigma*/
double generateSigma() {
	double sigmaMax = 1.0;
	double sigmaMin = 0.1;
	double sigma = sigmaMax - (sigmaMax - sigmaMin)*t;
	return sigma;
}

/*generate Gaussian random parameter*/
double generateGaussian(double mu, double sigma) {
	const double epsilon = numeric_limits<double>::min();
	const double two_pi = 2.0*3.14159265358979323846;

	static double z0, z1;
	static bool generate;
	generate = !generate;
	if (!generate) return z1 * sigma + mu;

	double u1, u2;
	do{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	}
	while (u1 <= epsilon);
	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}

void print() {
	cout //<< setiosflags(ios::left) << setw(3)
		<< T << " gBest fitness value: " << gBest.fitness;
	cout << " gBest value: ";
	for (auto i : gBest.POS) {
		cout //<< setiosflags(ios::fixed) << setprecision(5) << setiosflags(ios::left) << setw(8)
			<< i << " ";
	}
	cout << endl;
}