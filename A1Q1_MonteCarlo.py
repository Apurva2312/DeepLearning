import math
import random


# method to calculate area under shaded region/curve
def monte_carlo_shaded_area(r, num_of_points):
    number_of_points_under_curve = 0
    for j in range(num_of_points):
        # generate random points in the range 0 to r
        x = random.uniform(0, r)
        # generate random points in the range 0 to 2*(math.exp(r) (maximum side length of rectangle on y-axis)
        y = random.uniform(0, 2*(math.exp(r)))
        # if the point falls on or under the curve then increase the count
        if y <= math.exp(x):
            number_of_points_under_curve += 1

    # calculating area of rectangle = length(r) * breadth (2 * math.exp(r))
    area_of_rectangle = r * 2 * math.exp(r)
    area_under_curve = (number_of_points_under_curve / num_of_points) * area_of_rectangle
    return area_under_curve


# main method
if __name__ == '__main__':
    # total number of points to be generated randomly
    number_of_points = 10000
    for i in range(1, 11):
        shaded_area = monte_carlo_shaded_area(i, number_of_points)
        print("When r = ", i, ", estimated shadow area is: ", shaded_area)



