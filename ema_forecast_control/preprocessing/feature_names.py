# model features either used for preprocessing or model training

# Meta information
META = ['Participant', 'Trigger', 'Trigger_counter', 'Trigger_date', 'Form_start_date', 'Form_finish_date', 'Form_upload_date', 'Form',
        'interactive_start_time_pre', 'interactive_end_time_pre']

# Transformed Meta information
TRANSFORMED_META = ['Participant', 'DateTime', 'DayNr', 'Date', 'Time', 'Timerels', 'Form', 'Task', 
                    'DeliveryType', 'DeliveryProbability', 
                    'InterventionScore', 'Duration', 
                    'interactive_start_time_pre', 'interactive_end_time_pre']

# Observation space dimensions
OBSERVATION_FEAT = ['EMA_mood','EMA_disappointed','EMA_scared','EMA_worry',
'EMA_down','EMA_sad','EMA_confidence','EMA_emotion_control','EMA_stress','EMA_lonely',
'EMA_energetic','EMA_concentration','EMA_emotion_change','EMA_resilience','EMA_tired',
'EMA_satisfied', 'EMA_relaxed']

# Thereof conditional observations
CONDITIONAL_FEAT = ['EMA_emotion_control', 'EMA_emotion_change', 'EMA_sleep', 'EMA_joyful_day']

# Thereof non-conditional observations
NON_CONDITIONAL_FEAT = [s for s in OBSERVATION_FEAT if s not in CONDITIONAL_FEAT]

# Observation dimensions are incorporated in the score and weighted according to this
SCORE_WEIGHTS = {'EMA_mood': 1, 
                 'EMA_disappointed': 1,
                 'EMA_scared': 1,
                 'EMA_worry': 1,
                 'EMA_down': 1,
                 'EMA_sad': 1,
                 'EMA_confidence': 1,
                 'EMA_emotion_control': 0,
                 'EMA_stress': 1,
                 'EMA_lonely': 1,
                 'EMA_energetic': 1,
                 'EMA_concentration': 1,
                 'EMA_emotion_change': 0,
                 'EMA_resilience': 1,
                 'EMA_tired': 1,
                 'EMA_satisfied': 1,
                 'EMA_relaxed': 1}

# These EMAs become input to the PLRNN
INPUT_FEAT = ['EMA_sleep','EMA_joyful_day','EMA_feelactive_sincebeep','EMA_activity_pleas',
              'EMA_social_pleas','EMA_company','EMA_social_alone_yes','EMA_firstsignal']
              # 'EMA_activity_current', 'EMA_social']

# These EMAs become single input variables between -1 and 1
ORDINAL_DOMAIN_1_7 = ['EMA_sleep','EMA_joyful_day','EMA_feelactive_sincebeep']

# These EMAs become single input variables between -1 and 1, while their original interval is [-3, 3]
ORDINAL_DOMAIN_N3_3 = ['EMA_activity_pleas']

# These EMA sets are combined into single input variables between -1 and 1
COMBINE_ORDINAL_DOMAIN_1_7 = [['EMA_social_pleas','EMA_company']]

# These are the names of the combined input EMA
COMBINED_INPUT_NAMES = ['EMA_social_satisfied']

# These EMAs are transformed into binary one-hot inputs
BINARY_DOMAIN_1_2 = ['EMA_social_alone_yes']
BINARY_DOMAIN_0_1 = ['EMA_firstsignal']

# These EMAs are transformed into multi-category one-hot inputs (TODO: specify categories)
CATEGORICAL_DOMAIN = ['EMA_activity_current', 'EMA_social']

# These EMAs are formulated negatively and have to be recoded before anything
FLIP_OBSERVATION = ['EMA_disappointed','EMA_scared','EMA_worry',
'EMA_down','EMA_sad','EMA_stress','EMA_lonely','EMA_tired','EMA_concentration']

# These EMAs meant for input are formulated negatively and have to be recoded before anything
FLIP_INPUT = ['EMA_company']

# Combined list of EMAs interpreted as inputs
TRANSFORMED_INPUT_FEAT = (ORDINAL_DOMAIN_1_7
                            + ORDINAL_DOMAIN_N3_3
                            + COMBINED_INPUT_NAMES
                            + BINARY_DOMAIN_1_2
                            + BINARY_DOMAIN_0_1)

# Task names
INTERVENTION_NAMES_DE = ['Emotionaler Kompass', 'Den Atem zählen', 'Ruhiger, sicherer Ort', 'Atmen mit Pausen',
                      'Mitfühlender Begleiter', 'Emotionen als Welle', 'Tagebuch der Freudenmomente', 'Erfolgs-Logbuch']

# Task names English
INTERVENTION_NAMES_EN = ['Compass of emotions', 'Counting your breath', 'My calm and safe place', 'Breathing with breaks', 
 'My compassionate companion', 'Emotion as a wave', 'Journal of joyful moments', 'Positive data log']

# Interactive Task names
INTERACTIVE_NAMES = ['interactive1', 'interactive2', 'interactive3', 'interactive4',
                      'interactive5', 'interactive6', 'interactive7', 'interactive8']

# Consolidation Task names
CONSOLIDATION_NAMES = ['consolidation1', 'consolidation2', 'consolidation3', 'consolidation4',
                      'consolidation5', 'consolidation6', 'consolidation7', 'consolidation8']

DELIVERY_TYPES = ['NO REQUEST', 
                  'NO LOG', 
                  'APP (NO ANSWER)', 
                  'APP (TIMEOUT)', 
                  'APP (MODEL ERROR)'
                  'MISMATCH', 
                  'ML INFORMED', 
                  'RANDOM']

# Contructs
CONSTRUCTS_ALL = {
        'Positive affect': ['EMA_mood', 'EMA_relaxed', 'EMA_satisfied'],
        'Negative affect': ['EMA_scared', 'EMA_down', 'EMA_sad'],
        'Self-esteem': ['EMA_disappointed', 'EMA_confidence'],
        'Worrying': ['EMA_worry'],
        'Activity level': ['EMA_energetic', 'EMA_tired'],
        'Stress': ['EMA_stress', 'EMA_concentration', 'EMA_activity_pleas', 'EMA_social_pleas'],
        'Social isolation': ['EMA_lonely'],
        'Resilience': ['EMA_resilience'],
        'Emotion Regulation': ['EMA_emotion_control', 'EMA_emotion_change'],
                }

CONSTRUCTS_OBS = {k: vs for (k, vs) in CONSTRUCTS_ALL.items() if all([(v in OBSERVATION_FEAT) for v in vs])}

CONSTRUCTS_ANY_NON_CONDITIONAL = {k: vs for (k, vs) in CONSTRUCTS_ALL.items() if any([(v in NON_CONDITIONAL_FEAT) for v in vs])}
# CONSTRUCTS_INPUT = {k: vs for (k, vs) in CONSTRUCTS_ALL.items() if all([(v in INPUT_FEAT) for v in vs])}

GERMAN_EMA_LABELS = ['gute Stimmung',
                     'enttäuscht*',
                     'ängstlich*',
                     'nachdenklich*',
                     'niedergeschlagen*',
                     'traurig*',
                     'selbstbewusst',
                     'gestresst*',
                     'einsam*',
                     'voller Energie',
                     'unkonzentriert*',
                     'zuversichtlich',
                     'müde*',
                     'zufrieden',
                     'entspannt']