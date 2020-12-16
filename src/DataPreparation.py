class DataPreparation:
    data_set = []

    def __init__(self, dataSet):
        self.data_set = dataSet

    def set_feature_values(self, basic_G3=True, test=False):
        """
        Always add the full parameters like basic_G3=False.
        This will prevent you from changing self.
        parameters
        ----------
        basic_G3 : Boolean
            If True G3 feature is a Boolean, otherwise it's numerical.
        test: Boolean
            If true give the first 10 rows of the data set.
            This will decrease the time for testing new methods elsewhere.
        :rtype: object
        """
        self.data_set['school'].replace({'GP': 0, 'MS': 1}, inplace=True)
        self.data_set['sex'].replace({'F': 0, 'M': 1}, inplace=True)
        self.data_set['address'].replace({'U': 0, 'R': 1}, inplace=True)
        self.data_set['famsize'].replace({'LE3': 0, 'GT3': 1}, inplace=True)
        self.data_set['Pstatus'].replace({'T': 0, 'A': 1}, inplace=True)
        self.data_set['Mjob'].replace({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4},
                                      inplace=True)
        self.data_set['Fjob'].replace({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4},
                                      inplace=True)
        self.data_set['reason'].replace({'home': 0, 'reputation': 1, 'course': 2, 'other': 3}, inplace=True)
        self.data_set['guardian'].replace({'mother': 0, 'father': 1, 'other': 2}, inplace=True)
        self.data_set['schoolsup'].replace({'no': 0, 'yes': 1}, inplace=True)
        self.data_set['famsup'].replace({'no': 0, 'yes': 1}, inplace=True)
        self.data_set['paid'].replace({'no': 0, 'yes': 1}, inplace=True)
        self.data_set['activities'].replace({'no': 0, 'yes': 1}, inplace=True)
        self.data_set['nursery'].replace({'no': 0, 'yes': 1}, inplace=True)
        self.data_set['higher'].replace({'no': 0, 'yes': 1}, inplace=True)
        self.data_set['internet'].replace({'no': 0, 'yes': 1}, inplace=True)
        self.data_set['romantic'].replace({'no': 0, 'yes': 1}, inplace=True)
        # TODO:  Possibly 4 numerical values to see which students barely passed any see why.
        if basic_G3:
            self.data_set['G3'].replace({0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
                                         10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1},
                                        inplace=True)
        else:
            self.data_set['G3'].replace({0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
                                         10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3},
                                        inplace=True)
        if test:
            return self.data_set.head(10)
        else:
            return self.data_set
