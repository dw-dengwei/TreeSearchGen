def get_consistent_response(fn, args, keys, times=5, to_key=lambda x: x, to_value=lambda x: x, keep_structure=False):
	if isinstance(keys, str):
		keys = [keys]
	assert isinstance(keys, (list, tuple))
	assert callable(to_key) or isinstance(to_key, dict)
	assert callable(to_value) or isinstance(to_key, dict)

	answers = [fn(*args)['output'] for _ in range(times)]
	vote = {k: {} for k in keys}
	output = {k: None for k in keys}

	for k in keys:
		for ans in answers:
			if callable(to_key):
				tmp = to_key(ans[k])
			else:
				tmp = to_key[k](ans[k])

			if tmp not in vote[k].keys():
				vote[k][tmp] = 1
			else:
				vote[k][tmp] += 1
		output[k] = max(vote[k], key=vote[k].get)
		if callable(to_value):
			output[k] = to_value(output[k])
		else:
			output[k] = to_value[k](output[k])
	if len(keys) == 1:
		output = output[keys[0]]
	return output, answers
