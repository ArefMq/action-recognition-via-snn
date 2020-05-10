#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import os
import requests


_URL = os.environ.get('NOTIFY_URL', None)
_SLACK_NOTIFY = _URL is not None


def notify(*msgs, **kwargs):
    mark = kwargs.get('mark', None)
    title = kwargs.get('title', None)
    channel = kwargs.get('channel', None)
    icon = kwargs.get('icon', None)
    print_in_console = kwargs.get('print_in_console', True)

    if mark is None:
        pass
    elif mark == 'info':
        mark = 'ℹ️'
    elif mark == 'error':
        mark = '📛'
    elif mark == 'ok':
        mark = '✅'
    elif mark == 'warn' or mark == 'warning':
        mark = '⚠️'
    elif mark == 'heart':
        mark = '🧡'
    else:
        mark = '[%s]' % mark

    msg = '' if mark is None else ('%s ' % mark)
    msg += '  '.join([str(m) for m in msgs])
    if print_in_console:
        if title is not None:
            print('%s) \t%s' % (title, msg))
        else:
            print(msg)

    if not _SLACK_NOTIFY:
        return

    data = {
        'text': msg
    }
    if channel is not None:
        data['channel'] = '#%s' % channel
    if title is not None:
        data['name'] = title
    if icon is not None:
        data['icon'] = ':%s:' % icon

    try:
        requests.post(_URL, json=data)
    except:
        print('     can not write to channel...')



if __name__ == "__main__":
    notify('test message')
    notify('and a message with everything', mark='heart', icon='ghost', channel='general', title='wonderful')

