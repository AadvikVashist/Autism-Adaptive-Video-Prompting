import uuid
from typing import Union
import read_settings
from datetime import datetime, timedelta
import re
def create_uuid(increment = True):
    if increment:
        seq = read_settings.get_and_increment_sequence()
    else:
        seq = read_settings.get_sequence()
    u = uuid.uuid1(uuid.getnode(), clock_seq=seq)
    uuid_time = u.hex

    return uuid_time,u
def uuid_time(uuid : Union[str, uuid.UUID]):
    if type(uuid) is str:
        uuid = uuid.UUID(uuid)
    return datetime(1582, 10, 15) + timedelta(microseconds=uuid.time//10)
def get_data_from_string_uuid(cur_uuid : Union[str, uuid.UUID]):
    if type(cur_uuid) is str:
        cur_uuid = uuid.UUID(cur_uuid)
    time = uuid_time(cur_uuid)
    mac_adress = uuid.getnode()
    print (':'.join(re.findall('..', '%012x' % mac_adress)))
    sequence = 0
uuids, u = create_uuid(True)
get_data_from_string_uuid(u)
