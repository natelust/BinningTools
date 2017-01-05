#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <array>
#include <vector>
#include <string>

namespace py = pybind11;

#define DEBUG 0

namespace pyBinningTools {

class DownSample{
 public:
    DownSample(py::array_t<double, py::array::c_style | py::array::forcecast>
           input,
           int outWidth,
           int outHeight,
           int searchRad = 1, bool disjoint=false):_input(input),
                                               _width(outWidth),
                                               _height(outHeight),
                                               _searchRad(searchRad),
                                               _disjoint(disjoint) {
        _inputInfo = input.request();
        if (_inputInfo.ndim != 2) {
            throw std::runtime_error("Input must have two dimensions");
        }
        xFactor = static_cast<double>(_inputInfo.shape[1])/
                  static_cast<double>(_width);
        yFactor = static_cast<double>(_inputInfo.shape[0])/
                  static_cast<double>(_height);
    }
    py::array_t<double> operator()(std::vector<double> & verticies) {
        /* i and j are the coordinates in the reference frame of the input
         * ii, jj, guessY, guessX are coordinates in the reference frame of
         * the output image.
         */
         if (verticies.size() != ((_width+1) * (_height +1) * 2) && !_disjoint) {
             auto str = "Not enough input verticies, should be length output "
                        "width+1 * output height+1 times two "
                        "(verticies for both x and y)";
             throw std::runtime_error(str);
         } else if ( verticies.size() != (_width * 2) * (_height * 2) *2  && _disjoint) {
             auto str = "Not enough input verticies, should be length output "
                        "width*2 * output height*2 times two";
             throw std::runtime_error(str);
         }

        auto inputPtr = (double *) _inputInfo.ptr;

        // Initialize the return array (should be initialized to zero)
        auto output = py::array_t<double>(std::vector<std::size_t>{
                      static_cast<std::size_t>(_height),
                      static_cast<std::size_t>(_width)});
        auto outputInfo = output.request();
        auto outputPtr = (double *) outputInfo.ptr;
        for (int x = 0; x < _width*_height; ++x){
            outputPtr[x] = 0;
        }
        // Loop over each of the pixels in the saved PSF and determine
        // which bin the point should fall in
        for (int i = 0; i < _inputInfo.shape[0]; ++i) {
            for (int j = 0; j < _inputInfo.shape[1]; ++j) {
                double scaledY = static_cast<double>(i)/yFactor;
                double scaledX = static_cast<double>(j)/xFactor;
                // Find the naive guess about where the pixel will map to
                // and only search a few pixels around that position
                int guessY = static_cast<int>(floor(scaledY));
                int guessX = static_cast<int>(floor(scaledX));

                // debug parameter remove, or change to a ifdef
                bool notFound = true;

                // Check if the guess works outright, if not,
                // start a small searching loop
                if (pointInBin(scaledY, scaledX, guessY, guessX, verticies)) {
                    outputPtr[guessY * outputInfo.shape[0] + guessX] +=
                        inputPtr[i * _inputInfo.shape[0] + j];
                    notFound = false;
                } else {
                    for (int ii = guessY > 0 ? guessY - _searchRad: 0;
                         ii <= (guessY + _searchRad) &&
                         ii< (_height + 1) && notFound; ++ii) {
                        for (int jj = guessX > 0? guessX  - _searchRad: 0;
                             jj <= (guessX + _searchRad) &&
                             jj < (_width + 1)&& notFound; ++jj) {
                            // We have already tested the guessY and guessX
                            // position so just continue to the end of the loop
                            if (ii == guessY && jj == guessX) {
                                continue;
                            }
                            // If the point is contained in a given output bin,
                            // add the flux value to the output bin
                            if (pointInBin(scaledY, scaledX, ii, jj, verticies))
                            {
                                outputPtr[ii * outputInfo.shape[0] + jj] +=
                                inputPtr[i * _inputInfo.shape[0] + j];
                                notFound = false;
                            }
                        }
                    }
                }
                #if DEBUG > 0
                if (notFound) {
                    std::cout<< "The point at " << i << "," << j
                               << "was not added to output" << std::endl;
                }
                #endif
            }
        }
    return output;
    }

private:
    py::array_t<double> _input;
    py::buffer_info _inputInfo;
    int _width, _height, _searchRad;
    double xFactor, yFactor;
    bool _disjoint;

    bool pointInBin(double i, double j, int ii, int jj,
                    std::vector<double> & verticies){
        /* i and j are in the input reference frame
         * ii and jj are in the output array reference frame ii is y, jj is x
         */
        // Find the upper left coord in x, which will be the width in x + 1
        // times y coordinate (this is the number of memory offsets to get
        // to the begining of the row that contains the coordinate,), then
        // offset by the x coordinate
        // X bounds
        int lowerLeft, lowerRight, upperLeft, upperRight, offset;
        if (_disjoint) {
            ii *= 2;
            jj *= 2;
            lowerLeft = (_width*2)*(ii) + jj;
            lowerRight = (_width*2)*(ii) + jj + 1;
            upperLeft = (_width*2)*(ii+1) + jj;
            upperRight = (_width*2)*(ii+1) + jj + 1;
            offset = _width * 2 * _height *2;
        } else{
            lowerLeft = (_width+1)*(ii) + jj;
            lowerRight = (_width+1)*(ii) + jj + 1;
            upperLeft = (_width+1)*(ii+1) + jj;
            upperRight = (_width+1)*(ii+1) + jj + 1;
            offset = (_width + 1) * (_height + 1);
        }

        // Look up lambda caupture if need to be a reference
        auto getXVal = [&verticies](int n){ return verticies[n]; };
        auto getYVal = [&verticies, &offset](int n){
            return verticies[offset + n]; };

        // in the first two x is the indipendent variable, in the latter y is
        // the indipendent variable
        if (i >= line(j, getXVal(lowerLeft),
                         getYVal(lowerLeft),
                         getXVal(lowerRight),
                         getYVal(lowerRight)) &&

            i < line(j, getXVal(upperLeft),
                         getYVal(upperLeft),
                         getXVal(upperRight),
                         getYVal(upperRight)) &&

            j >= line(i, getYVal(lowerLeft),
                         getXVal(lowerLeft),
                         getYVal(upperLeft),
                         getXVal(upperLeft)) &&

            j < line(i, getYVal(lowerRight),
                         getXVal(lowerRight),
                         getYVal(upperRight),
                         getXVal(upperRight))) {
                return true;
        } else {
            return false;
        }

    }

    double line(double const & testPoint, double const & indepLeft,
                double const & depLeft, double const & indepRight,
                double const & depRight) {
        auto m = (depLeft - depRight) / (indepLeft - indepRight);
        auto b = depRight - (m * indepRight);
        return (m * testPoint + b);
    }

};
} // End namespace pyDownSample

PYBIND11_PLUGIN(binningExt) {
    py::module m("binningExt", "C++ python module for array binning functions");
    py::class_<pyBinningTools::DownSample> cls(m, "DownSample");

    cls.def(py::init<py::array_t<double, py::array::c_style |
                                 py::array::forcecast>,
                     int, int, int, bool>(), py::arg("input"),
           py::arg("outWidth"),
           py::arg("outHeight"), py::arg("searchRad") = 1,
           py::arg("disjoint") = false )
       .def("__call__", &pyBinningTools::DownSample::operator());

    return m.ptr();
}
