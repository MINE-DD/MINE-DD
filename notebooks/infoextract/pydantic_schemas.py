from pydantic import BaseModel, Field

class Measurement(BaseModel):
    measurement_type: str = Field(
        description="The type of measurement (e.g., Relative Humidity, Specific Humidity)."
    )
    description: str = Field(
        description="Description of how the measurement is computed or estimated."
    )
    formula: list[str] | None = Field(
        description="Formula(s) used for computation, if applicable."
    )
    variables: list[str] | None = Field(
        description="Variables used in the formulas."
    )
    notes: str | None = Field(
        description="Additional notes about the measurement, such as handling of missing data."
    )


class WeatherData(BaseModel):
    time_period: str | None = Field(
        description='Time in which weather data was collected, or time window in which environmental variables were monitored',
        examples=['Precipitation was collected during the monsoon season',
                  'monthly weather data were monitored between 2001 and 2012']
    )
    details: list[str] | None = Field(
        description='Any sentence from the paper describing any info related to weather data that cannot be captured well in the above fields',
        examples='Precipitations in Bangladesh were scarse during the 2005 fall'
    )
    measurement: list[Measurement] = Field(
        description="List of different measurements and methodologies used in the study."
    )


# %%
class Methodology(BaseModel):
    name: str | None = Field(
        description='Name of the methodology used',
        examples=['t-test', 'signal processing']
    )
    description: str | None = Field(
        description='Literal text, Description of the methodology'
    )
    methods: list[str] | None = Field(
        description='Specific methods or techniques used within the methodology',
        examples='Fast fourier transformation'
    )
    goals: list[str] | None = Field(
        description='Goals or purposes of using the methodology'
    )


class StudyScope(BaseModel):
    study_sites: list[str] | None = Field(
        description='Sites studied and mentioned in the paper',
        examples=['The Gambia', 'Mali']
    )
    description: list[str] | None = Field(
        description='Sites/setting description in sentences which do not necessarily contain the name of the site',
        examples=['resource limited setting']
    )
    locations: list[str] = Field(
        description='Names of the actual locations'
    )



class StudyDuration(BaseModel):
    duration: str | None = Field(
        description='The duration of the study described in the paper',
        examples=['three years', '6 months', 'the study ran over 10 years']
    )
    details: list[str] | None = Field(
        description='Actual sentences whereby the duration is mentioned. It is possible in this field to have multiple durations mentioned'
    )


class ParticipantAgeGroup(BaseModel):
    age_range: str | None = Field(
        description='Age range of the population described in the paper. If there is no range, also age in months/years is fine',
        examples=['36 months old', 'children younger than five years of age']
    )
    details: list[str] | None = Field(
        description='Any descriptive sentence that details who the participants were, how old they were and any other information that relates to them',
        examples=['Children 0–59 months of age with moderate-to-severe diarrhea (MSD)',
                  'Only the first 8–9 children in each age strata (0–11 months, 12–23 months, 24–59 months) were recruited']
    )


class Bibliography(BaseModel):
    article_type: str | None = Field(
        description='Article type',
        examples=['Research article', ' meta-analysis', 'review', 'opinion paper']
    )
    study_type: str | None = Field(
        description='Type of study',
        examples=['retrospective study']
    )
    title: str = Field(
        description='Title of the paper'
    )
    study_scope: StudyScope = []
    methodology: list[Methodology] = []
    study_duration: StudyDuration = []
    participant_age_group: ParticipantAgeGroup = []
    weather_data: WeatherData = []
    data_collection: str = Field(
        description='When data was collected?',
        examples=['during 2008 to 2011']
    )
    authors: str = Field(
        description='Authors of the paper. Usually listed below the title in the first page of the paper',
        examples=['Roose, A., Washington, D., Aniston, J.A.']
    )
    affiliation: str = Field(
        description='Affiliations of the authors listed in the authors field',
        examples=['Institute for Disease Modeling, Bellevue, Washington',
                  'International School for Advanced Studies, Trieste, Italy']
    )
    citation: str | None = Field(
        description='Citation of the paper which typically includes the first author surname followed by et al., and the year of publication along with the name of the journal',
        examples=['Roose et al., 2008, BMJ']
    )
    corresponding_author: str = Field(
        description='Author that has made available their email for further contact',
        examples=['r.roosevelt@gmail.com']
    )
    doi: str | None
    date: str = Field(
        description='Date in which the paper has been published comprising typically day, month and year',
        examples=['06-08-2013']
    )
    github_repo: str | None = Field(
        description='Github repository where code and data are available.',
        examples='https://github.com/papermanuscript/main'
    )
    journal: str = Field(
        description='Name of the journal in which the paper has been published, tipically listed in the front page, nearby the title',
        examples='PLoS Negl Trop Dis'
    )
