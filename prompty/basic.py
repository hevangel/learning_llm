import json
import prompty
# to use the openai invoker make 
# sure to install prompty like this:
# pip install prompty[openai]
import prompty.openai
from prompty.tracer import trace, Tracer, console_tracer, PromptyTracer

# add console and json tracer:
# this only has to be done once
# at application startup
Tracer.add("console", console_tracer)
json_tracer = PromptyTracer()
Tracer.add("PromptyTracer", json_tracer.tracer)

# if your prompty file uses environment variables make
# sure they are loaded properly for correct execution

def run(    
      firstName: any,
      context: any,
      question: any
) -> str:

  # execute the prompty file
  result = prompty.execute(
    "basic.prompty", 
    inputs={
      "firstName": firstName,
      "context": context,
      "question": question
    }
  )

  return result

if __name__ == "__main__":
   json_input = '''{
  "firstName": "Seth",
  "context": "The Alpine Explorer Tent boasts a detachable divider for privacy,  numerous mesh windows and adjustable vents for ventilation, and  a waterproof design. It even has a built-in gear loft for storing  your outdoor essentials. In short, it's a blend of privacy, comfort,  and convenience, making it your second home in the heart of nature!\\n",
  "question": "What can you tell me about your tents?"
}'''
   args = json.loads(json_input)

   result = run(**args)
   print(result)
