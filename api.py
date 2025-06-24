from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict # Ensure these are imported
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Car Price Prediction API! Use /predict_single, /compare_cars, or /market_analysis"}

# Mock prediction function
def predict_price(car_data: Dict) -> float:
    """Replace this with your actual model prediction"""
    base_price = 5.0  # Base price in lakhs
    
    # Simple mock logic
    price = base_price
    if car_data.get('fueltype') == 'Diesel': # Use .get() for safer access
        price += 0.5
    elif car_data.get('fueltype') == 'Electric':
        price += 1.0
        
    if car_data.get('transmission') == 'Automatic':
        price += 0.3
        
    # Depreciation
    # Ensure 'year' and 'kilometersdriven' exist before using
    year = car_data.get('year', 2023) 
    kilometersdriven = car_data.get('kilometersdriven', 0)

    price -= (2023 - year) * 0.2
    price -= kilometersdriven / 100000
    
    return max(price, 0.5)  # Minimum price

class CarData(BaseModel):
    name: Optional[str] = None # Added default None
    location: str
    year: int
    kilometersdriven: int
    fueltype: str
    transmission: str
    ownertype: str
    mileage: Optional[float] = None
    engine: Optional[float] = None
    power: Optional[float] = None
    seats: Optional[float] = None

# MarketAnalysisData can be identical to CarData or have specific fields if needed
class MarketAnalysisData(BaseModel):
    name: Optional[str] = None
    location: str
    year: int
    kilometersdriven: int
    fueltype: str
    transmission: str
    ownertype: str
    mileage: Optional[float] = None
    engine: Optional[float] = None
    power: Optional[float] = None
    seats: Optional[float] = None


@app.post("/predict_single")
async def predict_single(car: CarData):
    try:
        logger.info(f"Received data for single prediction: {car.dict()}")
        prediction = predict_price(car.dict())
        return {"prediction": round(prediction, 2)}
    except Exception as e:
        logger.error(f"Error during single prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to predict price: {str(e)}")

@app.post("/compare_cars")
async def compare_cars(cars: List[CarData]):
    try:
        if len(cars) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 cars for comparison")
            
        results = []
        for car in cars:
            prediction = predict_price(car.dict())
            results.append({
                **car.dict(), # Spread the car data
                "prediction": round(prediction, 2)
            })
        
        prices = [c['prediction'] for c in results]
        
        # Ensure numpy functions handle empty lists gracefully if 'prices' could be empty (though it shouldn't here)
        average_price = round(np.mean(prices), 2) if prices else 0
        max_price = round(max(prices), 2) if prices else 0
        min_price = round(min(prices), 2) if prices else 0
        price_range = round(max_price - min_price, 2) if prices else 0

        return {
            "cars": results,
            "stats": {
                "average": average_price,
                "max": max_price,
                "min": min_price,
                "range": price_range
            }
        }
    except Exception as e:
        logger.error(f"Error during car comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to compare cars: {str(e)}")

@app.post("/market_analysis")
async def market_analysis(car: MarketAnalysisData):
    try:
        logger.info(f"Received market analysis data: {car.dict()}")
        
        # 1. Get the base prediction for the given car
        base_prediction = predict_price(car.dict())

        # 2. Simulate market position (this is mock logic, replace with real analysis)
        market_position = "around the"
        # Example: if the base prediction is above a certain threshold for its class/year
        if base_prediction > 8.0: # Arbitrary threshold for demonstration
            market_position = "above the"
        elif base_prediction < 4.0: # Arbitrary threshold for demonstration
            market_position = "below the"

        # 3. Simulate scenarios (mock variations)
        scenarios = []

        # Scenario 1: Better condition (fewer KM driven)
        better_km_car = car.dict()
        # Ensure kilometersdriven is not negative
        better_km_car['kilometersdriven'] = max(0, car.kilometersdriven - 15000) 
        price_better_km = predict_price(better_km_car)
        diff_better_km = round(price_better_km - base_prediction, 2)
        percent_diff_better_km = round((diff_better_km / base_prediction) * 100, 2) if base_prediction != 0 else 0
        scenarios.append({
            "variation": "If 15,000 KM less driven",
            "price": round(price_better_km, 2),
            "difference": diff_better_km,
            "percent_diff": percent_diff_better_km
        })

        # Scenario 2: Newer year
        newer_year_car = car.dict()
        newer_year_car['year'] = min(2025, car.year + 1) # Max year 2025, current year is 2025
        price_newer_year = predict_price(newer_year_car)
        diff_newer_year = round(price_newer_year - base_prediction, 2)
        percent_diff_newer_year = round((diff_newer_year / base_prediction) * 100, 2) if base_prediction != 0 else 0
        scenarios.append({
            "variation": "If 1 year newer",
            "price": round(price_newer_year, 2),
            "difference": diff_newer_year,
            "percent_diff": percent_diff_newer_year
        })
        
        # Scenario 3: Different owner type (e.g., if First instead of Second)
        if car.ownertype == "Second":
            first_owner_car = car.dict()
            first_owner_car['ownertype'] = "First"
            price_first_owner = predict_price(first_owner_car)
            diff_first_owner = round(price_first_owner - base_prediction, 2)
            percent_diff_first_owner = round((diff_first_owner / base_prediction) * 100, 2) if base_prediction != 0 else 0
            scenarios.append({
                "variation": "If First Owner",
                "price": round(price_first_owner, 2),
                "difference": diff_first_owner,
                "percent_diff": percent_diff_first_owner
            })


        return {
            "base_price": round(base_prediction, 2),
            "market_position": market_position,
            "scenarios": scenarios
        }

    except Exception as e:
        logger.error(f"Error during market analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze market position: {str(e)}")