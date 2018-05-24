import requests

# from .credentials import FOOTBALL_DATA_API_TOKEN
FOOTBALL_DATA_API_TOKEN = 'b1c19af0135842d0a08759c45d0c2de7'


class DataAccessLayer:
    def __init__(self):
        self._token = FOOTBALL_DATA_API_TOKEN

    @staticmethod
    def _kwargs_to_query(**kwargs):
        _arr = []
        for k, v in kwargs.items():
            if v:
                _arr.append(f'{k}={v}')
        _query = '?' + '&'.join(_arr) if len(_arr) > 0 else ''
        return _query

    def _get_data(self, namespace, response_control='full'):
        headers = {'X-Auth-Token': self._token, 'X-Response-Control': response_control}
        return requests.get(f'http://api.football-data.org/v1/{namespace}', headers=headers).json()

    # competition stats
    def get_competitions(self, season=None):
        """

        :param season: String /\d\d\d\d/
            Defaults to the current year, given as 4 digit like '2015'.
        :return:
        """
        _query = self._kwargs_to_query(season=season)
        return self._get_data(f'competitions/{_query}')

    def get_competition_teams(self, competition_id):
        return self._get_data(f'competitions/{competition_id}/teams')

    def get_league_table(self, competition_id, matchday=None):
        """
        :param competition_id:
        :param matchday: Integer /[1-4]*[0-9]*/
            For the leageTable subresource, the matchday defaults to the current matchday.
            For former seasons the last matchday is taken.
            For the fixture resource, it's unset.
        :return:
        """
        _query = self._kwargs_to_query(matchday=matchday)
        return self._get_data(f'competitions/{competition_id}/leagueTable{_query}')

    def get_competition_fixtures(self, competition_id, timeframe=None, matchday=None):
        _query = self._kwargs_to_query(timeFrame=timeframe, matchday=matchday)
        return self._get_data(f'competitions/{competition_id}/fixtures{_query}')

    # team stats
    def get_team(self, team_id):
        return self._get_data(f'teams/{team_id}')

    def get_team_players(self, team_id):
        return self._get_data(f'teams/{team_id}/players')

    def get_team_fixtures(self, team_id, season=None, timeframe=None, venue=None):
        """

        :param team_id:
        :param season: String /\d\d\d\d/
            Defaults to the current year, given as 4 digit like '2015'.
        :param timeframe: p|n[1-9]{1,2}
            The value of the timeFrame argument must start with either p(ast) or n(ext),
            representing a timeframe either in the past or future.
            It is followed by a number in the range 1..99.
            It defaults to n7 in the fixture resource and is unset for fixture as a subresource.
            For instance: p6 would return all fixtures in the last 6 days,
                whereas n23 would result in returning all fixtures in the next 23 days.
        :param venue: String /away|home/
            Defines the venue of a fixture. Default is unset and means to return all fixtures.
        :return:
        """
        _query = self._kwargs_to_query(season=season, timeFrame=timeframe, venue=venue)
        return self._get_data(f'teams/{team_id}/fixtures{_query}')

    # fixtures
    def get_fixture(self, fixture_id, head2head=None):
        """

        :param fixture_id:
        :param head2head: Integer /[0-9]+/
            Define the the number of former games to be analyzed in the head2head node. Defaults to 10.
        :return:
        """
        _query = self._kwargs_to_query(head2head=head2head)
        return self._get_data(f'fixtures/{fixture_id}/{_query}')

    def get_all_fixtures(self, timeframe, league):
        """

        :param timeframe:
        :param league: (comma separated) String /[\w\d]{2,4}(,[\w\d]{2,4})*/
            A (list of, comma separated) league-code(s).
            Default is unset and means all available.
            See the competition list resource for currently available leagues
            or the appendix of the full documentation for a table of all league codes

        :return:
        """
        _query = self._kwargs_to_query(timeFrame=timeframe, league=league)
        return self._get_data(f'fixtures/{_query}')


if __name__ == '__main__':
    from pprint import pprint
    # resp = DataAccessLayer().get_league_table(398, matchday=1)
    # resp = DataAccessLayer().get_team_players(66)
    resp = DataAccessLayer().get_competition_fixtures(398, matchday=1)
    pprint(resp)
