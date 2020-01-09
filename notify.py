#!/usr/bin/env python
# -*- coding: utf-8 -*- 

def notify(*msgs, **kwargs):
    mark = kwargs.get('mark', None)
    title = kwargs.get('title', None)
    channel = kwargs.get('channel', None)
    icon = kwargs.get('icon', None)

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
    if title is not None:
        print('%s) \t%s' % (title, msg))
    else:
        print(msg)


if __name__ == "__main__":
    notify('test message')
    notify('and a message with everything', mark='heart', icon='ghost', channel='general', title='wonderful')

