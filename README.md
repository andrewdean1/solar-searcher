# Project Solar Searcher
-   Noah Price
-   Omar Armbruster
-   Andrew Dean
-   Col McDermott

## Abstract
As the severity of climate change continues to increase and the consumption of energy in the US remains massive, a critical focus of today’s power industry is establishing sources of renewable energy to replace the less climate-friendly alternatives.  Three of the most common renewable energy solutions are solar, hydro, and wind-powered generators.  While these renewable energy sources are proven to be both highly effective and regenerative, they often require a complex set of specific circumstances for construction, (such as geographical location/elements and consistent weather conditions) to function effectively and comparably to traditional, environmentally abrasive methods.  

Focussing on solar energy, we aim to address the problem of locating optimal regions throughout the US to build solar farms for cleaner energy production.  Our general approach to this problem is to determine the predicted energy output of (and possibly corresponding implementation cost) installing a solar farm at a given US location.  Interpreting location-specific characteristics extracted from geographical, topographical, and meteorological (weather) data, we aim to calculate the projected energy production from placing a solar farm (or similar solar-powered operation) anywhere in the US.  

Our primary goal is to leverage several machine learning techniques to assist our success, converting this problem into a regression task with a predictive model.  To measure and evaluate our progress on this project, we strive to implement (most likely employing pre-existing software libraries) and refine a regression model that processes the aforementioned data of a given input and predicts the potential energy production of placing solar energy structures at any analyzed region.  We intend to create a reproducible data processing procedure, construct a usable model, and provide an early-stage tool to help inform and promote decisions about implementing solar energy solutions in the optimal US regions.

## Motivation & Driving Questions
Establishing renewable and regenerative energy sources is a necessary step in combating global warming.  Our primary motivation for this project stems from our interest in applying our new knowledge of machine learning capabilities and techniques to seek solution steps for a crucial problem surrounding the world.  Our main plan for acquiring data is to combine geographical data from sources like Google Earth as well as the NASA Power Project with large-region weather data (sources yet to be identified). 

There are three underlying questions driving our project:  At this US location, what is predicted energy production (in standard power units) if solar panels are installed?  In general, what US regions will yield the highest solar energy production?  Considering the implementation cost and comparison to existing power industry structures, should solar energy structures be built at this US location/region?

We envision our predictive regression model being used to inform decisions regarding the installment of solar energy systems at various places throughout the US.  Our model could assist in deciding if the implementation of solar power on new, untouched US land (pertaining to energy production systems) is worthwhile or if existing power production methods in certain US regions should be converted into solar energy systems.  Additionally, if the time and scope of this project allow, we could extend our model to be used for similar decisions regarding the other primary renewable energy sources like hydro and wind power.

## Planned Deliverables 
We plan to deliver a python package (available on Github) containing trained models to predict the potential energy output at any given latitude and longitude in the US for all three of our energy sources. We also plan to deliver a heatmap showing the locations of maximum and minimum energy outputs. We also would like to use our model and associated cost data to find the optimal locations to build new solar, wind, or hydro farms.  

In the event that we are unable to complete each of our goals, we plan to scale down the project by just modeling one or two of the renewable energy sources. We are confident that we can at least create a model for the potential of solar energy sources and will expand to wind and hydro if time allows. If this is the case, we would still like to create a heatmap for our modeled resources, but may not be able to complete the optimization step where we determine the most efficient/cost effective method at a given location.


## Resources Required 
The primary resource required will be data regarding the three renewable energy sources for various locations in the U.S. We have already identified potential data sources from the [NASA Power Project](https://registry.opendata.aws/nasa-power/#:~:text=The%20POWER%20project%20contains%20over,resolution%20of%20the%20source%20products) and a [Github repository](https://github.com/Charlie5DH/Solar-Power-Datasets-and-Resources) and will explore further as needed. We will likely need to create an account with the [Google Maps API](https://developers.google.com/maps/documentation/elevation/overview) in order to easily pull needed information (elevation, terrain type, etc.) for making predictions on new locations. Because we are working with large datasets (and potentially a neural network), we may require additional computing power to train our models in a timely manner. We plan to use the research computers available to physics and computer science departments in this scenario.

## What We Will Learn
Col: As most of the learning in class thus far has focussed on classification, I’m excited to learn more about regression tasks and the corresponding ML models/techniques.  In addition to this, I’m looking forward to learning more about and gaining more hands-on experience scraping data from public online sources.  I feel that these learning opportunities will develop my understanding of how ML theory and abstract software tools can be harnessed to tackle complex, empirical problems.

Omar: I’m planning on learning more about data processing and cleaning (as we likely will need to compile data from a variety of sources) as well as learning how to pull data from the google API. It will be interesting to explore packages like GeoPandas for creating our heatmaps and using our model to predict energy output at locations without any existing data.

Noah: I’m looking forward to learning about algorithms and data pipelines as solutions to complex problems. In particular, I’m interested to see how our work on solar energy can be re-tooled or otherwise translated to perform predictions in the fields of wind or hydro energy. I’m also excited to see how we can take topographical data and translate it into effective features.

Andrew: I’m excited to deepen my expertise in optimization methods as they apply to renewable energy. Specifically, it’ll be interesting to delve into how diverse geographical features – such as elevation gradients, land use patterns, and climatic variabilities – affect solar energy potential. This project will also offer me the opportunity to integrate advanced data visualization techniques, furthering my ability to interpret and communicate complex data. I also hope to bridge the gap between theoretical predictions and practical usage (ex. site selection), ultimately contributing something meaningful in the renewables space.

## Risk Statement
The most significant potential obstacle in our path to successfully implementing this project is the data. In order to make accurate and meaningful predictions about potential energy output for a theoretical plant, we will need to develop a strong understanding of the factors at play in generating renewable energy. To account for this difficulty, we have already identified several potential datasets, but we have also discussed multiple options for applicable models in this scenario. The optimal features may not be the same for all possible models, so if our data is not yielding the patterns we expected, we can be flexible and experiment with different means of prediction.

Another potential obstacle is compute power. Given that we plan to include a heatmap in our final deliverable, we will need to apply our model to a very wide dataset. To account for this problem, we have access to powerful lab computers in both the computer science and physics departments, on which we can run our most intensive experiments to generate our final reports.

## Ethics Statement
Our model tackles a subject which is often politicized, as different parties hold different views on climate change and the best approach to stabilizing global resources. Given the importance of developing renewable energy sources, our model does have the potential to cause harm if it exhibits bias. If our model over-predicts energy output in certain areas and under-predicts output in others, application of our results could yield unequal distribution of green energy, which may not be the optimal solution to our energy crisis. Further, if our model predicts overall low outputs, its results could theoretically be used to justify propagation of traditional energy methods such as fossil fuels. 

However, we overall find that our model has the potential to do good, under the following assumptions:
Renewable energy output can be predicted based on topographical, geological, and other spatial features.
Developing theoretical means of measurement for green energy will make its implementation more robust.
The world is a better place when our energy sources are not harmful to the planet.

## Tentative Timeline
By Week 3:
- Establish a working data acquisition and processing pipeline from NASA, GitHub, Kaggle, and Google Earth sources.
- Complete initial exploratory data analysis and feature extraction to validate data quality and refine our goals. 
- Develop a basic regression model prototype that can predict solar energy output on a small, representative dataset – demonstrating “something that works.”

By Week 6:
- Refine and scale up the regression model using the full dataset, integrating additional relevant features such as elevation and terrain type from Google Maps.
- Produce preliminary heatmaps to visualize predicted energy outputs across selected regions of the US.
- Potentially begin incorporating solar analysis with wind and hydro energy data to evaluate optimal energy sources
- Or, further scale and solidify the usage of our solar model and build a simple interactive app, allowing users to input parameters (for example, lat and long coordinates) and instantly visualize tailored energy output predictions


This phased approach will ensure that by the Week 9/10 checkpoint, we have a robust, functioning system that not only demonstrates early success but also provides clear direction for the final refinements before our Week 12 presentations.

## Required File Structure

To run the code in this repository, you will need a .env file in the Data Sources directory with the following structure:

```
GOOGLE_MAPS_API_KEY = [your google maps API key]
```
