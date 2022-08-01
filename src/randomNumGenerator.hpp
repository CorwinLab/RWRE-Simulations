#ifndef RANDOMNUMGENERATOR_HPP_
#define RANDOMNUMGENERATOR_HPP_

#include <boost/math/distributions.hpp>
#include <boost/random.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <random>

class RandomNumGenerator{
    protected:
        double beta;
        boost::random::beta_distribution<>::param_type betaParams;
        std::random_device rd;
        boost::random::mt19937_64 gen;
        std::uniform_real_distribution<> dis;
        boost::random::beta_distribution<> beta_dist;

    public: 
        RandomNumGenerator(const double _beta);
        ~RandomNumGenerator(){};

        double getBeta() { return beta; };
        void setBeta(double _beta) { beta = _beta; };
        double generateBeta();
        void setBetaSeed(const unsigned int seed) { gen.seed(seed); };
};

#endif /* RANDOMNUMGENERATOR_HPP_ */