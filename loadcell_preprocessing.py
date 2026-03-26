import numpy as np

class LoadCellProcessor:
    def __init__(self, tolerance=5.0):
        self.tare_value = 0.0
        self.last_weight = 0.0
        self.tolerance = tolerance # Delta for outlier rejection
    
    def set_tare(self, raw_value):
        """
        Calculates the weight of the "empty" system to subtract from future readings.
        """
        self.tare_value = raw_value
        print(f"✅ Software Tare Set: {self.tare_value:.2f}g")

    def preprocess_weight(self, raw_value):
        """
        Implements Load Cell Data Preprocessing as per the workflow diagram:
        1. Software Tare subraction
        2. Outlier Rejection (Simple delta check or range check)
        """
        # 1. Apply Tare
        tared_weight = raw_value - self.tare_value
        
        # 2. Outlier Rejection (e.g. discard negatives or extreme noise)
        # If it's a huge jump from the last reading (assuming 1Hz sample rate), reject it.
        # But if it's the first reading, just accept.
        if abs(tared_weight - self.last_weight) > 50.0 and self.last_weight != 0:
            # Rejection logic: return last known good value
            return self.last_weight
            
        # Range check: Bottles shouldn't weigh more than 1kg (1000g)
        if tared_weight < 0:
            return 0.0
        elif tared_weight > 1000.0:
            return self.last_weight # Use last good value if it's a spike
            
        self.last_weight = tared_weight
        return tared_weight

if __name__ == "__main__":
    # Test block
    processor = LoadCellProcessor()
    processor.set_tare(10.5) # System weighs 10.5g when empty
    
    print(f"Weight 20.5: {processor.preprocess_weight(20.5):.2f}g (Expected: 10.0)")
    print(f"Weight 10000: {processor.preprocess_weight(10000.0):.2f}g (Expected: Outlier Rejected -> 10.0)")
