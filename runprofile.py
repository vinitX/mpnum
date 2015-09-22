# encoding: utf-8


import cProfile


def main():
    cProfile.run('import runtests; runtests.main()')


if __name__ == '__main__':
    main()


