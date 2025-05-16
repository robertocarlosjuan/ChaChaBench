# "Ihlen_0_int":                [("move", "down")],
# "Merom_1_int":                [("move", "down"), ("turn", "roll_left")],
# grocery_store_half_stocked("turn", "yaw_right")],
# "hall_glass_ceiling":         [("turn", "pitch_down")],
# "hotel_gym_spa":              [("move", "down")

# "office_large":          move down -> pitch black
# Office_large move right

# Restaurant_asian move up -> pitch black
# Restaurant_diner roll right
# Wainscott_1_int ("turn", "yaw_left")
# generate for above scenes
scene2command_map ={
   "office_large":               [("move", "right")], 
   "Merom_1_int":                [("move", "down")],
   "Ihlen_0_int":                [("move", "down")],
   "Beechwood_1_int":            [("move", "up")],

}
# scene2command_map ={
#     "Beechwood_1_int":            [("move", "up")],
#     "Ihlen_0_int":                [("move", "down")],
#     "Merom_1_int":                [("move", "down"), ("turn", "roll_left")],
#     "grocery_store_half_stocked": [("turn", "yaw_right")],
#     "hall_glass_ceiling":         [("turn", "pitch_down")],
#     "hotel_gym_spa":              [("move", "down")],
#     "office_large":               [("move", "down"), ("move", "right")],
#     "restaurant_asian":           [("move", "up")],
#     "restaurant_diner":           [("turn", "roll_right")],
#     "Wainscott_1_int":            [("turn", "yaw_left")],
# }
# scene2command_map = {
#     "Beechwood_0_int":            [("move", "left")],
#     "Beechwood_1_int":            [("move", "up"), ("turn", "yaw_left")],

#     "Benevolence_1_int":          [("move", "left"), ("move", "up")],
#     "Benevolence_2_int":          [("move", "right"), ("move", "up"), ("turn", "yaw_right")],

#     "Ihlen_0_int":                [("move", "down")],
#     "Ihlen_1_int":                [("move", "up")],

#     "Merom_0_int":                [("move", "right")],
#     "Merom_1_int":                [("move", "down"), ("turn", "roll_left")],

#     "Pomaria_0_garden":           [("turn", "roll_right")],
#     "Pomaria_0_int":              [("turn", "yaw_left")],
#     "Pomaria_1_int":              [("move", "up"), ("turn", "yaw_left")],
#     "Pomaria_2_int":              [("move", "left"), ("turn", "yaw_right"), ("move", "up")],

#     "grocery_store_asian":        [("move", "right")],
#     "grocery_store_convenience":  [("turn", "yaw_right")],
#     "grocery_store_half_stocked": [("turn", "roll_left"), ("turn", "roll_right"), ("turn", "yaw_right")],

#     "hall_glass_ceiling":         [("turn", "pitch_down")],

#     "hotel_gym_spa":              [("move", "down"), ("move", "right"), ("move", "up")],
#     "hotel_suite_large":          [("turn", "roll_right"), ("turn", "yaw_left")],
#     "hotel_suite_small":          [("move", "up"), ("turn", "pitch_up")],

#     "house_double_floor_upper":   [("move", "left"), ("turn", "pitch_down")],

#     "office_bike":                [("move", "right")],
#     "office_cubicles_left":       [("move", "backward"), ("move", "right")],
#     "office_large":               [("turn", "pitch_down"), ("move", "down"),
#                                    ("move", "forward"), ("move", "right")],

#     "restaurant_asian":           [("move", "backward"), ("move", "forward"), ("move", "up")],
#     "restaurant_brunch":          [("move", "down")],
#     "restaurant_diner":           [("move", "backward"), ("turn", "roll_right")],
#     "restaurant_urban":           [("turn", "yaw_left")],
#     "Rs_int":                     [("turn", "roll_right")],  
#     "Rs_garden":                  [("turn", "yaw_right"), ("turn", "roll_right")],
#     "restaurant_hotel":           [("turn", "roll_left"), ("turn", "pitch_up")],
#     "Wainscott_1_int":            [("move", "down"), ("move", "right"), ("turn", "pitch_up"), ("turn", "yaw_left")],
#     "school_computer_lab_and_infirmary": [("turn", "roll_right")],
#     "restaurant_brunch":          [("move", "down"), ("turn", "roll_left")],
#     "school_geography":           [("move", "forward"), ("move", "left"), ("turn", "roll_right")],
#     "restaurant_cafeteria":       [("move", "backward")],
#     "school_chemistry":           [("turn", "roll_left"), ("turn", "yaw_left")],
    
# }

total = sum(len(v) for k, v in scene2command_map.items())
print(f" total regenerating commands= {total}")
print(f" total scenes for which we have regenerating commands= {len(scene2command_map)}")




# moving objects
# restaurant_diner_MB1p14_0

# doable
# Wainscott_1_int_MD1p18_0.mp4
# Wainscott_1_int_MR0p69_0
# Wainscott_1_int_TPU1p07_0
# Wainscott_1_int_TYL2p24_0
# Rs_int_TRR3p14_0
# restaurant_hotel_TRL2p56_0

# spawn outside
# school_computer_lab_and_infirmary_TRR2p45_0

# dark
# restaurant_brunch_MD1p45_0
# restaurant_brunch_TRL2p21_0

# Very hard
# school_geography_MF1p04_0
# Rs_garden_TYR1p23_0

# pitch white
# school_geography_ML0p94_0
# restaurant_cafeteria_MB0p53_0
# restaurant_cafeteria_MB0p53_0

# black:
# school_chemistry_TRL1p66_0
# school_chemistry_TYL1p59_0
# Rs_garden_TRR2p82_0
# restaurant_urban_TYL1p06_0
# restaurant_brunch_MD1p45_0