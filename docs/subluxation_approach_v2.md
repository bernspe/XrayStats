# Analysis proposal for subluxation detection

## Rationale
When scanning xrays for vertebral subluxation phenomena, which is nothing else then leaving a smoothened curve which rises from the bottom vertebrae,
we look for deviation behaviour of single vertebrae. Sometimes multiple vertebrae can be shifted, which can be called a staircase phenomenon.

## Technical approach
 - use the spline generated from nodes as ground truth
 - smoothen the spline, but keep the ends in place
 - recalculate the shortest distances for each point on the smoothened spline 
   - to nearby nodes
   - to nearby midpoints of segmental lines from Cobb_segmental
 - map those measurements to the vertebra
 - so we should get a record like `{vertebra: [node distance, lower segmental line distance, upper segmental line distance]}à
 - e.g. a classical L4/5 Pseudolisthesis should result in a higher deviation for L4 than for L3 or L5

## Model
Try to find a discriminatory model, e.g. logistic regression, k-Means, random forest,... to detect subluxation levels.
You can use /Users/peterbernstein/Python/XrayStats/data/subluxation_feedback.csv as train/test data. Sublux levels will be given 
as `upper vertebra/lower vertebra`, which is a segmental notation.

## Segmental notation for subluxation
 - `upper vertebra/lower vertebra`
 - assume that the lower vertebra is the one
   - which follows the smoothened line more than the upper vertebra
   - with lower distances than the upper vertebra in the following parameters
     - node distance
     - upper segmental line distance
 - assume that the upper vertebra is the one
   - that is unstable
   - with distances above a certain threshold (which should be normalized against the calibration line) in the following parameters
     - node distance
     - lower segmental line distance