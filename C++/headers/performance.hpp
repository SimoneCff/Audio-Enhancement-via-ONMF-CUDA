#ifndef PERFORMANCE_HPP
#define PERFORMANCE_HPP

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

class Performance_Writer
{
private:
    ofstream csv_file;
    double tempo_creazione_dict;
    double tempo_estrazione;
    double tempo_totale;

    void write_header()
    {
        if (csv_file.is_open()) {
            csv_file << "nome_file,tempo_creazione_dict,tempo_estrazione,tempo_totale\n";
        }
    }

public:
    Performance_Writer(const string& filename)
    {
        bool fileHasData = false;
        {
            ifstream checkFile(filename);
            if (checkFile.good() && checkFile.peek() != EOF) {
                fileHasData = true;
            }
        }
        csv_file.open(filename, ios::out | ios::app);
        if (!csv_file.is_open()) {
            cerr << "Impossibile aprire il file CSV: " << filename << endl;
        }
        if(!fileHasData) {
            write_header();
        }
    }

    void set_tempo_creazione_dict(double tempo_creazione_dict)
    {
        this->tempo_creazione_dict = tempo_creazione_dict;
    }

    void set_tempo_estrazione(double tempo_estrazione)
    {
        this->tempo_estrazione = tempo_estrazione;
    }

    void set_tempo_totale(double tempo_totale)
    {
        this->tempo_totale = tempo_totale;
    }

    void append_record(const string& nome_file)
    {
        if (csv_file.is_open()) {
            csv_file << nome_file << ","
                     << tempo_creazione_dict << ","
                     << tempo_estrazione << ","
                     << tempo_totale << "\n";
        }
    }

    ~Performance_Writer()
    {
        if (csv_file.is_open()) {
            csv_file.close();
        }
    }
};

#endif // PERFORMANCE_HPP