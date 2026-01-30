#ml 

# Support

$$ support(I) = \frac{\text {\# transactions containing I}}{\text{\# transactions}}$$

# Confidence

$$confidence (I1 ->I2) = \frac{\text{\# transactions containing I1 and I2}}{\text {\# transactions containing I1}}$$

# Lift

$$lift(I1 -> I2) = \frac{\text {confidence (I1 -> I2)}}{\text {support (I2)}}$$
Lift essentially improves the confidence by normalizing against the likelihood of I2 in the overall population.


# Apriori Algorithm

1. Set a minimum support and confidence.
2. Take all the subsets in transactions having higher support than the minimum support.
3. Take all the rules of these subsets having higher confidence than the minimum confidence.
4. Sort the rules by decreasing lift.