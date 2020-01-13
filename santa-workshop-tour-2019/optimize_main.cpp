// https://www.kaggle.com/golubev/c-stochastic-product-search-65ns
// g++ -pthread -lpthread -O3 -std=c++17 -o optimize_main optimize_main.cpp
#include <cstring>
#include <array>
#include <cassert>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <random>
using namespace std;
#include <chrono>
using namespace std::chrono;


// You can setup how many families you need for swaps and what best choice use for each family 
constexpr array<uint8_t, 14> DISTRIBUTION{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5}; 
//constexpr array<uint8_t, 16> DISTRIBUTION{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 5}; 
// {2, 5} it's mean the first random family will brute force for choices 1-2 and the second random family will brute force for choices 1-5 

constexpr int MAX_OCCUPANCY = 300;
constexpr int MIN_OCCUPANCY = 125;
constexpr int BEST_N = 1000;
array<uint8_t, 5000> n_people;
array<array<uint8_t, 10>, 5000> choices;
array<array<uint16_t, 10>, 5000> PCOSTM;
array<array<double, 176>, 176> ACOSTM;

void init_data() {
    ifstream in("./atad/family_data.csv");
    
    assert(in && "family_data.csv");
    string header;
    int n,x;
    char comma;
    getline(in, header);
    for (int j = 0; j < choices.size(); ++j) {
        in >> x >> comma;
        for (int i = 0; i < 10; ++i) {
            in >> x >> comma;
            choices[j][i] = x-1;
        }
        in >> n;
        n_people[j] = n;
    }
    array<int, 10> pc{0, 50, 50, 100, 200, 200, 300, 300, 400, 500};
    array<int, 10> pn{0,  0,  9,   9,   9,  18,  18,  36,  36, 235};
    for (int j = 0; j < PCOSTM.size(); ++j)
        for (int i = 0; i < 10; ++i)
            PCOSTM[j][i] = pc[i] + pn[i] * n_people[j];
    
    for (int i = 0; i < 176; ++i)
        for (int j = 0; j < 176; ++j)
            ACOSTM[i][j] = i * pow(i+125, 0.5 + abs(i-j) / 50.0) / 400.0;
}
array<uint8_t, 5000> read_submission(string filename) {
    ifstream in(filename);
    //assert(in && "submission.csv");
    assert(in);
    array<uint8_t, 5000> assigned_day{};
    string header;
    int id, x;
    char comma;
    getline(in, header);
    for (int j = 0; j < choices.size(); ++j) {
        in >> id >> comma >> x;
        assigned_day[j] = x-1;
        auto it = find(begin(choices[j]), end(choices[j]), assigned_day[j]);
        if (it != end(choices[j]))
            assigned_day[j] = distance(begin(choices[j]), it);
    }
    return assigned_day;
}
struct Index {
    Index(array<uint8_t, 5000> assigned_days_) : assigned_days(assigned_days_)  {
        setup();
    }
    array<uint8_t, 5000> assigned_days;
    array<uint16_t, 100> daily_occupancy_{};
    int preference_cost_ = 0;
    void setup() {
        preference_cost_ = 0;
        daily_occupancy_.fill(0);
        for (int j = 0; j < assigned_days.size(); ++j) {
            daily_occupancy_[choices[j][assigned_days[j]]] += n_people[j];
            preference_cost_ += PCOSTM[j][assigned_days[j]];
        }
    }
    double calc(const array<uint16_t, 5000>& indices, const array<uint8_t, DISTRIBUTION.size()>& change) {
        double accounting_penalty = 0.0;
        auto daily_occupancy = daily_occupancy_;
        int preference_cost = preference_cost_;
        for (int i = 0; i < DISTRIBUTION.size(); ++i) {
            int j = indices[i];
            daily_occupancy[choices[j][assigned_days[j]]] -= n_people[j];
            daily_occupancy[choices[j][       change[i]]] += n_people[j];
            
            preference_cost += PCOSTM[j][change[i]] - PCOSTM[j][assigned_days[j]];
        }

        for (auto occupancy : daily_occupancy)
            if (occupancy < MIN_OCCUPANCY)
                return 1e12*(MIN_OCCUPANCY-occupancy);
            else if (occupancy > MAX_OCCUPANCY)
                return 1e12*(occupancy - MAX_OCCUPANCY);

        for (int day = 0; day < 99; ++day)
            accounting_penalty += ACOSTM[daily_occupancy[day]-125][daily_occupancy[day+1]-125];

        accounting_penalty += ACOSTM[daily_occupancy[99]-125][daily_occupancy[99]-125];
        return preference_cost + accounting_penalty;
    }
    void reindex(const array<uint16_t, DISTRIBUTION.size()>& indices, const array<uint8_t, DISTRIBUTION.size()>& change) {
        for (int i = 0; i < DISTRIBUTION.size(); ++i) {
            assigned_days[indices[i]] = change[i];
        }
        setup();
    }
};

double calc(const array<uint8_t, 5000>& assigned_days, bool print=false) {
    int preference_cost = 0;
    double accounting_penalty = 0.0;
    array<uint16_t, 100> daily_occupancy{};
    for (int j = 0; j < assigned_days.size(); ++j) {
        preference_cost += PCOSTM[j][assigned_days[j]];
        daily_occupancy[choices[j][assigned_days[j]]] += n_people[j];
    }
    for (auto occupancy : daily_occupancy)
        if (occupancy < MIN_OCCUPANCY)
            return 1e12*(MIN_OCCUPANCY-occupancy);
        else if (occupancy > MAX_OCCUPANCY)
            return 1e12*(occupancy - MAX_OCCUPANCY);

    for (int day = 0; day < 99; ++day)
        accounting_penalty += ACOSTM[daily_occupancy[day]-125][daily_occupancy[day+1]-125];

    accounting_penalty += ACOSTM[daily_occupancy[99]-125][daily_occupancy[99]-125];
    if (print) {
        cout << preference_cost << " " << accounting_penalty << " " << preference_cost+accounting_penalty << endl;
    }
    return preference_cost + accounting_penalty;
}

void save_sub(const array<uint8_t, 5000>& assigned_day, double score) {
    cout << "saved a new best file: " << score << endl;
    char filename[500];
    sprintf(filename, "./mission/random_best/submission_%.3f.csv", score);
    // ofstream out("./mission/random_best/submission.csv");
    ofstream out(filename);
    out << "family_id,assigned_day" << endl;
    for (int i = 0; i < assigned_day.size(); ++i)
        out << i << "," << choices[i][assigned_day[i]]+1 << endl;
}
        
const vector<array<uint8_t, DISTRIBUTION.size()>> changes = []() {
    vector<array<uint8_t, DISTRIBUTION.size()>> arr;
    array<uint8_t, DISTRIBUTION.size()> tmp{};
    for (int i = 0; true; ++i) {
        arr.push_back(tmp);
        tmp[0] += 1;
        for (int j = 0; j < DISTRIBUTION.size(); ++j)
            if (tmp[j] >= DISTRIBUTION[j]) {
                if (j >= DISTRIBUTION.size()-1)
                    return arr;
                tmp[j] = 0;
                ++tmp[j+1];
            }
    }
    return arr;
}();

template<class ExitFunction>
void stochastic_product_search(Index index, ExitFunction fn) { // 15'360'000it/s  65ns/it  0.065µs/it
    double best_local_score = calc(index.assigned_days);
    thread_local std::mt19937 gen(std::random_device{}());
    uniform_int_distribution<> dis(0, 4999);
    array<uint16_t, 5000> indices;
    iota(begin(indices), end(indices), 0);
    array<uint16_t, DISTRIBUTION.size()> best_indices{};
    array<uint8_t, DISTRIBUTION.size()> best_change{};
    bool found_better_global = false;
    for (; fn();) {
        bool found_better = false;
        for (int k = 0; k < BEST_N; ++k) {
            for (int i = 0; i < DISTRIBUTION.size(); ++i) //random swap
                swap(indices[i], indices[dis(gen)]);
            for (const auto& change : changes) {
                auto score = index.calc(indices, change);
                if (score < best_local_score) {
                    found_better = true;
                    found_better_global = true;
                    best_local_score = score;
                    best_change = change;
                    //save_sub(index.assigned_days, score);
                    copy_n(begin(indices), DISTRIBUTION.size(), begin(best_indices));
                }
            }
        }
        if (found_better) { // reindex from N best if found better
            index.reindex(best_indices, best_change);
            double current_best_score = calc(index.assigned_days);
            save_sub(index.assigned_days, current_best_score);
            calc(index.assigned_days, true);
            
        }
    }
    if (found_better_global) {
        // save_sub(index.assigned_days);
    }
}

int main() {
    cout << "hello world" << 3.3333 << endl;
    //return 0;

    // cout << "saved a new best file" << endl;
    // char filename[50];
    //float score = 2222.2;
    //sprintf(filename, "./mission/random_best/submission_%.2f.csv", score);
    // ofstream out("./mission/random_best/submission.csv");
    //ofstream out(filename);
    //out << "family_id,assigned_day" << endl;

    init_data();
    //auto assigned_day = read_submission("./mission/submission_best_69880.40.csv");
    //auto assigned_day = read_submission("./atad/sample_submission.csv");
    //auto assigned_day = read_submission("./mission/submission_69761.84.csv");
    //auto assigned_day = read_submission("./mission/submission_69752.88.csv");
    //auto assigned_day = read_submission("./mission/submission_69740.834.csv");
    //auto assigned_day = read_submission("./mission/submission_69733.050.csv");
    //auto assigned_day = read_submission("./mission/submission_69723.152.csv");
    //auto assigned_day = read_submission("./mission/submission_69721.468.csv");
    //auto assigned_day = read_submission("./mission/submission_69695.779.csv");
    //auto assigned_day = read_submission("./mission/submission_69674.730.csv");
    auto assigned_day = read_submission("./mission/submission_69668.783.csv");

    Index index(assigned_day);
    calc(index.assigned_days, true);
//    auto forever = []() { return true; };
//    auto count_exit = [start = 0]() mutable { return (++start <= 1000); };
    auto time_exit = [start = high_resolution_clock::now()]() {
        // return duration_cast<minutes>(high_resolution_clock::now()-start).count() < 535; //8h55
        return duration_cast<minutes>(high_resolution_clock::now()-start).count() < 120000; //8h55
    };
    
    stochastic_product_search(index, time_exit);
    return 0;
}
