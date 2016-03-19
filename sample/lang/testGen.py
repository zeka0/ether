def test_gen(cycle):
    for i in xrange(cycle):
        yield i

if __name__ == '__main__':
    y = []
    y.extend(test_gen(10))
    y.extend(test_gen(1))
    y.append(test_gen(1))
    print y