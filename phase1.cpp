#include<mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "genann.h"
#include <conio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <queue>
#include <regex>
#include <math.h>
#include <time.h>
#include <omp.h>

using namespace std;
double start_time, end_time;


int samples = 1115;
int limitTR = round(0.7 * 1593);
int limitTS = 1593 - limitTR;
double TrainSet[1115][256] = { 0 };
double TRresult[1115] = { 0 };
double TSresult[478] = { 0 };
double TestSet[478][256] = { 0 };

int Tokenize()
{
	int sh0 = 0, sh1 = 0;
	int OT = 0;
	ifstream fin;
	string line;
	queue <string> q;
	string s, str;
	int CL = 0, count = 0;
	int X = 0;
	float NT = 0.7;
	vector<string> tokens;

	string address = "C:\\genann-master\\genann-master\\semeion.data";
	fin.open(address);
	if (fin.is_open())
	{
		while (getline(fin, line))
		{
			stringstream ss(line);
			q.push(ss.str());
		}
	}
	fin.close();

	cout << "The size of queue is: " << q.size() << "\n";
	count = 0;


		for (int i = 0; i < 1593; i++)
		{
			cout << "line  " << count << "\n";
			str = q.front();
			q.pop();

			regex reg("\\s+");
			sregex_token_iterator iter(str.begin(), str.end(), reg, -1);
			sregex_token_iterator end;
			vector<string> tokens(iter, end);
		
			CL = 0;
			for (const string &s : tokens) {

				CL = CL + 1;
				X = stoi(s);
				if (count < limitTR)
				{

					if (CL < 257)
					{
				
						TrainSet[count][CL] = X;
					}
					else if (X == 1) {
						
						OT = (CL - 257);
						TRresult[count] = OT;
					
					}
				}
				else {
					sh1 = sh1 + 1;
					if (CL < 257)
					{
						TestSet[count - limitTR][CL] = X;
					}
					else if (X == 1) {
					
						OT = (CL - 257);
						TSresult[count] = OT;
				
					}
				}
		
			}
			count = count + 1;
		}

	
	cout << "number of itteration is :" << sh0 << "   and    " << sh1 << endl;

	return 0;
}

int correct_predictions(genann *ann) {
	int correct = 0, j = 0;
	for (j = 0; j < samples; ++j)
	{
		const double *guess = genann_run(ann, TrainSet[j]);
		double max = 0.0;
		int k = 0, actual = 0, max_cls = 0;
		for (k = 0; k < 10; k++)
		{
			if (guess[k]> max) {
				max = guess[k];
				max_cls = k;
			}
			if (TSresult[j * 10 + k] == 1.0) actual = k;
		}
		
		if (TSresult[j * 10 + (int)max_cls] == 1.0) ++correct;
		
	}
	return correct;
}


int main(int argc, char *argv[])
{
	int i, j;
	int NUM_THREADS = 8;
	int iterations = 500;
	int rank, size;
	MPI_Status status;
	MPI_Request request;

	char address[] = "C:\\genann-master\\genann-master\\semeion.data";
	

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int loops = round(iterations / (size));

	genann *ann = genann_init(16 * 16, 1, 28, 10);

	if (rank == 0) {

		Tokenize();

		start_time = MPI_Wtime();

		

		MPI_Bcast(&loops, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&TrainSet, 1115*256, MPI_INT, 0, MPI_COMM_WORLD);
		
		

		genann *ann = genann_init(16 * 16, 1, 28, 10);
		#pragma omp parallel for
		
			for (int i = 0; i < loops; ++i) {
				cout << " iteration " << i << endl;
				for (int j = 0; j < 1115; ++j) {
					genann_train(ann, TrainSet[j], TRresult + j, .1);
				}
			}
		

		
	}
	if (rank > 0) {

		MPI_Bcast(&loops, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&TrainSet, 1115 * 256, MPI_INT, 0, MPI_COMM_WORLD);

		genann *ann = genann_init(16 * 16, 1, 28, 10);
		#pragma omp parallel for
		
			for (int i = 0; i < loops; ++i) {
				cout << " iteration " << i << endl;
				for (int j = 0; j < 1115; ++j) {
					genann_train(ann, TrainSet[j], TRresult + j, .1);
				}
			}
		
	
	}
	

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, ann->weight, ann->total_weights, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		for (j = 0; j<ann->total_weights; j++) { ann->weight[j] = ann->weight[j] / size; }
		MPI_Barrier(MPI_COMM_WORLD);

		if (rank == 0) {
			end_time = MPI_Wtime();
			printf("\ntrain time taken :  = %f\n\n", end_time - start_time);
			
			cout << "prediction start to run" << endl;
			int correct = correct_predictions(ann);
			printf("\n\n %d/%d accuracy (%0.1f%%).\n", correct, 478, (double)correct / 478 * 100.0);
		}
	
	MPI_Finalize();
	genann_free(ann);
	

	_getch();
	return 0;
}



