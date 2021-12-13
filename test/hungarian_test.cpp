#include <iostream>
#include <loco_framework/hungarian.hpp>

int main()
{
    std::cout << "HungarianAssignment test" << std::endl;
    // std::vector<std::vector<double> > distances = { { 10, 19, 8, 15, 0 }, 
    //                                                 { 10, 18, 7, 17, 0 }, 
    //                                                 { 13, 16, 9, 14, 0 }, 
    //                                                 { 12, 19, 8, 18, 0 } };
    std::vector<std::vector<double> > distances = { { 9999.0, 1.0, 0.5 }, 
                                                    { 9999.0, 6.0, 4.5 } };

    loco::HungarianAssignment hungarian(distances);
    std::vector<size_t> assignment;

    double total_cost = hungarian.assign(assignment);

    std::cout << std::endl << "Distance matrix: " << std::endl;
    for (const auto& v : distances)
    {
        for (const auto& d : v) std::cout << d << "\t";
        std::cout << std::endl;
    }

    std::cout << std::endl << "Assignment: " << std::endl;
    for (size_t i = 0; i < assignment.size(); i++)
    {
        std::cout << i << " --> " << assignment[i] << std::endl;
    }
    std::cout << std::endl << "Total cost: " << total_cost << std::endl;

    return 0;
}
