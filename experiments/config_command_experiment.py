from model_analyzer.config.input.config_command_profile import ConfigCommandProfile
from model_analyzer.config.input.config_field import ConfigField
from model_analyzer.config.input.config_primitive import ConfigPrimitive


class ConfigCommandExperiment(ConfigCommandProfile):
    """ 
    Extended ConfigCommandProfile with extra options for experiment algorithm configuration
    """

    def _fill_config(self):
        super()._fill_config()
        self._add_config(
            ConfigField('radius',
                        field_type=ConfigPrimitive(int),
                        flags=['--radius'],
                        default_value=2,
                        description='The size of the neighborhood radius'))
        self._add_config(
            ConfigField('magnitude',
                        field_type=ConfigPrimitive(int),
                        flags=['--magnitude'],
                        default_value=2,
                        description='The size of each step'))
        self._add_config(
            ConfigField(
                'min_initialized',
                field_type=ConfigPrimitive(int),
                flags=['--min-initialized'],
                default_value=2,
                description=
                'The minimum number of datapoints needed in a neighborhood before stepping'
            ))
