class Parameters:
    def __init__(self, values, index=0):
        self._values = values if isinstance(values, list) else [values]
        self._index = index

    @property
    def value(self):
        return self._values[self._index]

    def __iter__(self):
        return iter(self._values)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"FlexibleValue({self._values})"
    
    def __getitem__(self, index):
        return self._values[index]
    
    def __len__(self):
        return len(self._values)
    
    @staticmethod
    def get_values(min, max=None, scale="lin", number=1):
        if max is None:
            return Parameters(min)
        if scale == "lin":
            return Parameters([min + (max - min) * i / (number - 1) for i in range(number)])
        elif scale == "log":
            if min <= 0:
                raise ValueError("Logarithmic scale requires min > 0")
            return Parameters([min * (max / min) ** (i / (number - 1)) for i in range(number)])
        elif scale == "exp":
            if min <= 0:
                raise ValueError("Exponential scale requires min > 0")
            return Parameters([min * (max / min) ** (i / (number - 1)) for i in range(number)])
        else:
            raise ValueError("Invalid scale type. Use 'lin', 'log', or 'exp'.")
    
    @staticmethod
    def get_dubble_values(min1, min2, max1=None, max2=None, scale="lin", number=1):
        if max1 is None:
            return (min1, min2)
        values1 = Parameters.get_values(min1, max1, scale, number)
        values2 = Parameters.get_values(min2, max2, scale, number)
        return Parameters([(values1[i], values2[i]) for i in range(number)])

planner_type = 1  # 1 for Standard Trajectory Planner, 2 for MILP Trajectory Planner

num_obstacles = Parameters.get_values(4)

tau = Parameters.get_values(0.2)

horizon = Parameters.get_values(5)

iteration = Parameters.get_values(20)

obstacle_deviation = False
obstacle_deviations_values = Parameters.get_dubble_values(0.1, 0.05)
obstacle_deviations_times = 1

plot_trajectory = False

print(f"Parameters:\n"
      f"planner_type: {planner_type}\n"
      f"num_obstacles: {num_obstacles}\n"
      f"tau: {tau}\n"
      f"horizon: {horizon}\n"
      f"iteration: {iteration}\n"
      f"obstacle_deviation: {obstacle_deviation}\n"
      f"obstacle_deviations_values: {[x for x in obstacle_deviations_values]}\n"
      f"obstacle_deviations_times: {obstacle_deviations_times}\n"
      f"plot_trajectory: {plot_trajectory}")


